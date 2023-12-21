import logging

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINEConv, BatchNorm, LayerNorm

from .config import TURLGNNConfig
from ..turl.config import TURLConfig
from ..turl.model import TURL, TableEmbeddings, TableLayerSimple, TableMLMHead
from ..gnn.model import GINe, PNA, LinkPredHead
from ..gnn.config import GNNConfig

# Setup logging
logger = logging.getLogger(__name__)


class TURLGNN(nn.Module):
    def __init__(self, config: TURLGNNConfig, turl_config: TURLConfig, gnn_config: GNNConfig):
        super(TURLGNN, self).__init__()

        self.config = config

        # TURL
        self.turl = TURL(turl_config)
        
        # GNN
        if config.include_gnn_node_feats:
            # Create MLP to mix node features and entity embeds
            # n_input = gnn_config.n_node_feats + gnn_config.n_node_feats
            n_input = turl_config.hidden_size + gnn_config.n_node_feats
            n_hidden = 2 * n_input
            self.gnn_node_feat_mlp = nn.Sequential(
                BatchNorm(n_input),
                nn.Linear(n_input, n_hidden), nn.LeakyReLU(), nn.Dropout(config.mlp_dropout), 
                nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(), nn.Dropout(config.mlp_dropout),
                nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(), nn.Dropout(config.mlp_dropout),
                nn.Linear(n_hidden, gnn_config.n_node_feats))
        if config.gnn_arch == "gine":
            self.gnn = GINe(gnn_config)
        elif config.gnn_arch == "pna":
            self.gnn = PNA(gnn_config)

    def load_turl_pretrained(self, checkpoint):
        self.turl.load_pretrained(checkpoint)

    def load_gnn_pretrained(self, checkpoint):
        self.turl.load_pretrained(checkpoint)

    def freeze_turl(self):
        # Freeze the weights
        for param in self.turl.parameters():
            param.requires_grad = False

    def freeze_gnn(self):
        # Freeze the weights
        for param in self.gnn.parameters():
            param.requires_grad = False

    def forward(self, turl_kwargs, gnn_kwargs, seed_edge_node_ids, node_ent_ids):
        # Forward pass on TURL
        turl_outputs = self.turl(**turl_kwargs)

        # Feed entity embeddings into GNN
        id_ent_mask = turl_kwargs['id_ent_mask']
        seed_ents = turl_kwargs['input_ent'][0, id_ent_mask[0]]
        gnn_kwargs = self.prepare_gnn_input(gnn_kwargs, seed_ents, node_ent_ids)

        # Forward pass on GNN
        gnn_outputs, _ = self.gnn(**gnn_kwargs)
        
        return turl_outputs, gnn_outputs

    def get_ent_embeds(self, idx):
        return self.turl.table.embeddings.ent_embeddings(idx)
    
    def prepare_gnn_input(self, gnn_kwargs, seed_ents, node_ent_ids):
        node_feats_x = gnn_kwargs.pop('x')
        x = node_feats_x
        # Initialize node embeddings to that from the TURL entity vocabulary
        if self.config.node_feat_type == "vocab":
            #                       do not backprop vvvvvvvv
            x = self.get_ent_embeds(node_ent_ids[0]).detach()
            # node_ent_ids.shape = (batch_size, nb_nodes)
            # NOTE: We backprop ONLY on seed embeddings (!)
            seed_node_mask = torch.isin(node_ent_ids[0], seed_ents)
            x[seed_node_mask] = self.get_ent_embeds(node_ent_ids[0, seed_node_mask])
        # Combine node embedding with node features through an MLP
        if self.config.include_gnn_node_feats:
            x = self.gnn_node_feat_mlp(torch.cat((x, node_feats_x), dim=-1))
        gnn_kwargs['x'] = x

        return gnn_kwargs

    def gnn_loss_fn(self, *args, **kwargs):
        return self.gnn.loss_fn(*args, **kwargs)


class GNNLayer(nn.Module):
    def __init__(self, gnn_config: GNNConfig):
        super(GNNLayer, self).__init__()
        
        self.agg = GINEConv(nn.Sequential(
                    nn.Linear(gnn_config.n_hidden, gnn_config.n_hidden), 
                    nn.ReLU(), 
                    nn.Linear(gnn_config.n_hidden, gnn_config.n_hidden)
                ), edge_dim=gnn_config.n_hidden)
        self.batch_norm = BatchNorm(gnn_config.n_hidden)

    def forward(self, x, edge_index, edge_attr):
        return self.batch_norm(self.agg(x, edge_index, edge_attr))

class FusedTURLGNN(nn.Module):
    def __init__(self, 
                 config: TURLGNNConfig, 
                 turl_config: TURLConfig,
                 gnn_config: GNNConfig,
                 pooling_function: str = "mean"):
        super(FusedTURLGNN, self).__init__()

        self.config = config 
        self.turl_config = turl_config
        self.gnn_config = gnn_config

        self.layer_types = []
        self.layers = None
        self.mix_modules = None
        self.construct_arch()

        # TURL input embeddings
        # Entity vocab: self.turl_embeds.ent_embeddings
        self.turl_embeds = TableEmbeddings(turl_config)

        # GNN input embeddings
        self.gnn_node_embed = nn.Linear(turl_config.hidden_size, gnn_config.n_hidden)
        self.gnn_edge_embed = nn.Linear(gnn_config.n_edge_feats, gnn_config.n_hidden)
        if config.include_gnn_node_feats:
            # Create MLP to mix node features and entity embeds
            n_input = turl_config.hidden_size + gnn_config.n_node_feats
            n_output = turl_config.hidden_size
            n_hidden = 2 * n_input
            self.gnn_node_feat_mlp = nn.Sequential(
                BatchNorm(n_input),
                nn.Linear(n_input, n_hidden), nn.LeakyReLU(), nn.Dropout(config.mlp_dropout), 
                nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(), nn.Dropout(config.mlp_dropout),
                nn.Linear(n_hidden, n_hidden), nn.LeakyReLU(), nn.Dropout(config.mlp_dropout),
                nn.Linear(n_hidden, n_output))
        
        # How to pool embeddings
        if pooling_function == "mean":
            self.pooling = lambda E: torch.mean(torch.stack(E), dim=0)

        # Redout heads
        self.readout_gnn = LinkPredHead(gnn_config)
        self.readout_lm = TableMLMHead(turl_config)

    def construct_arch(self):
        self.layer_types.clear()
        self.layers = nn.ModuleList()
        self.mix_modules = nn.ModuleList()

        n_mix_input = self.turl_config.hidden_size + self.gnn_config.n_hidden
        n_mix_hidden = 2*n_mix_input

        for fused_layer in self.config.fused_arch_layers:
            typ, module_types = self.get_layers_and_type(fused_layer)
            layer_modules = self.layers_to_modulelist(module_types)
            # How to inter-mix LM and GNN embeddings
            mix_module = nn.Sequential(
                BatchNorm(n_mix_input),
                nn.Linear(n_mix_input, n_mix_hidden), nn.LeakyReLU(), nn.Dropout(self.config.mlp_dropout), 
                nn.Linear(n_mix_hidden, n_mix_hidden), nn.LeakyReLU(), nn.Dropout(self.config.mlp_dropout),
                nn.Linear(n_mix_hidden, n_mix_input))
                
            self.layer_types.append((typ, module_types))
            self.layers.append(layer_modules)
            self.mix_modules.append(mix_module)
        
    def get_layers_and_type(self, fused_layer: str):
        # Parallel
        if '|' in fused_layer:
            return '|', fused_layer.split('|')
        # Single layer
        return None, [fused_layer]
    
    def layers_to_modulelist(self, layers: list[str]):
        layer_map = {
            'lm': lambda: TableLayerSimple(self.turl_config),
            'gnn': lambda: GNNLayer(self.gnn_config),
        }
        return nn.ModuleList([layer_map[l]() for l in layers])

    def get_turl_embeds(self, turl_kwargs):
        # Validate input
        assert turl_kwargs['input_tok'].min() >= 0, \
          f'Max token ID is too small (!): {turl_kwargs["input_tok"].min()} < 0'
        assert turl_kwargs['input_tok'].max() < self.turl_embeds.word_embeddings.num_embeddings, \
          f'Max token ID is too large (!): {turl_kwargs["input_tok"].max()} >= {self.turl_embeds.word_embeddings.num_embeddings}'
        return self.turl_embeds(turl_kwargs['input_tok'], turl_kwargs['input_tok_type'], turl_kwargs['input_tok_pos'], 
                        turl_kwargs['input_ent'], turl_kwargs['input_ent_type'], turl_kwargs['ent_candidates'], turl_kwargs['input_ent_pos']) 

    def get_gnn_embeds(self, turl_kwargs, node_ent_ids, node_feats_x):
        #                                                   vvvvvvvvv do not backprop
        x = self.turl_embeds.ent_embeddings(node_ent_ids[0]).detach()
        # If we do not use a node vocab, there is no need to backprop
        if self.config.no_node_vocab:
            # NOTE: We backprop ONLY on seed embeddings (!)
            id_ent_mask = turl_kwargs['id_ent_mask']
            seed_node_mask = torch.isin(node_ent_ids[0], turl_kwargs['input_ent'][0, id_ent_mask[0]])
            x[seed_node_mask] = self.turl_embeds.ent_embeddings(node_ent_ids[0, seed_node_mask])
        # Combine node embedding with node features through an MLP
        if self.config.include_gnn_node_feats:
            x = self.gnn_node_feat_mlp(torch.cat((x, node_feats_x), dim=-1))
        return x

    def get_turl_attention_masks(self, turl_kwargs):
        extended_input_tok_mask, extended_input_ent_mask = None, None
        if turl_kwargs['input_tok_mask'] is not None:
            extended_input_tok_mask = turl_kwargs['input_tok_mask'][:, None, :, :]
            extended_input_tok_mask = extended_input_tok_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_input_tok_mask = (1.0 - extended_input_tok_mask) * -10000.0
        if turl_kwargs['input_ent_mask'] is not None:
            extended_input_ent_mask = turl_kwargs['input_ent_mask'][:, None, :, :]
            extended_input_ent_mask = extended_input_ent_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_input_ent_mask = (1.0 - extended_input_ent_mask) * -10000.0
        return extended_input_tok_mask, extended_input_ent_mask

    def pool_lm_seed_nodes(self, ent_hidden_states, seed_edge_node_ids, id_ent_mask):
        seed_x = ent_hidden_states[0, id_ent_mask[0]]
        # Pool embeddings from the same seed node into a unique embedding
        node_unique_ids, node_ids_where = torch.unique(seed_edge_node_ids.to(seed_x.device), return_inverse=True)
        node_embeds_pooled = []
        for i in range(len(node_unique_ids)):
            pooled = seed_x[node_ids_where == i].mean(dim=0)
            node_embeds_pooled.append(pooled)
        node_embeds_pooled = torch.stack(node_embeds_pooled)
        # node_unique_ids (shape=(n_of_seed_nodes,)): mapping of seed node in GNN's x node embeds array
        return node_unique_ids, node_ids_where, node_embeds_pooled

    def get_turl_outputs(self, tok_hidden_states, ent_hidden_states, ent_candidates_embeddings, turl_kwargs):
        ent_candidates = turl_kwargs['ent_candidates']
        tok_prediction_scores, ent_prediction_scores = self.readout_lm(tok_hidden_states, ent_hidden_states, ent_candidates, ent_candidates_embeddings)

        tok_masked_lm_labels, ent_masked_lm_labels, edge_feat_ent_mask, id_ent_mask, node_feat_ent_mask = \
            turl_kwargs['tok_masked_lm_labels'], turl_kwargs['ent_masked_lm_labels'], \
            turl_kwargs['edge_feat_ent_mask'], turl_kwargs['id_ent_mask'], turl_kwargs['node_feat_ent_mask']

        # Header token loss
        if tok_masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            if (tok_masked_lm_labels!=-1).int().sum() == 0:
                tok_masked_lm_loss = torch.tensor(0)
            else:
                tok_masked_lm_loss = loss_fct(tok_prediction_scores.view(-1, self.turl_config.vocab_size), tok_masked_lm_labels.view(-1))
            tok_outputs = (tok_masked_lm_loss, tok_prediction_scores)
            
        # Cell entity loss
        if ent_masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            # `ent_masked_lm_labels` contains the entity IDs as labels.
            # BUT, it should really just be the size of the `max_entity_candidate`,
            # so we have to match the entity IDs from the label with the IDs in the
            # entity candidates
            is_candidate = torch.zeros((*ent_masked_lm_labels.shape, ent_candidates.shape[-1])).bool()
            is_candidate = is_candidate.to(ent_masked_lm_labels.device)
            for batch_i in range(ent_candidates.shape[0]):
                is_candidate[batch_i] = torch.eq(ent_masked_lm_labels[batch_i, :, None], ent_candidates[batch_i])
            labels = is_candidate.int().argmax(dim=-1)

            # `ent_masked_lm_labels` contains the entity IDs (from the global entity vocab). 
            #     `labels` contains the id in the entity candidate list
            preds = ent_prediction_scores.view(-1, self.turl_config.max_entity_candidate)
            if edge_feat_ent_mask is None:
                ent_masked_loss = loss_fct(preds, labels.view(-1))
                edge_ent_loss = ent_masked_loss
                node_ent_loss = 0
                id_ent_loss = 0
            else:
                # edge_feat_ent_mask.shape = (batch_size, # entities)
                id_mask = id_ent_mask.view(-1)
                edge_mask = edge_feat_ent_mask.view(-1)
                node_mask = node_feat_ent_mask.view(-1)
                if id_mask.int().sum() == 0:
                    id_ent_loss = 0
                else:
                    id_ent_loss = loss_fct(preds[id_mask], labels.view(-1)[id_mask])
                if edge_mask.int().sum() == 0:
                    edge_ent_loss = 0
                else:
                    edge_ent_loss = loss_fct(preds[edge_mask], labels.view(-1)[edge_mask])
                if node_mask.int().sum() == 0:
                    node_ent_loss = 0
                else:
                    node_ent_loss = loss_fct(preds[node_mask], labels.view(-1)[node_mask])
            ent_outputs = ((id_ent_loss, edge_ent_loss, node_ent_loss), ent_prediction_scores)
        return tok_outputs, ent_outputs

    def get_gnn_outputs(self, x, gnn_kwargs):
        pos_edge_index, neg_edge_index = gnn_kwargs['pos_edge_index'], gnn_kwargs['neg_edge_index']
        neg_edge_attr = self.gnn_edge_embed(gnn_kwargs['neg_edge_attr'])
        pos_edge_attr = self.gnn_edge_embed(gnn_kwargs['pos_edge_attr'])
        out = self.readout_gnn(x, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr)
        return out
    
    def forward(self, turl_kwargs, gnn_kwargs, seed_edge_node_ids, node_ent_ids):
        """Forward function.
        
        Arguments
        ---------
        seed_edge_node_ids : tensor
            Shape is `(n_seed_edges*2)`. This tensor contains node indices. The index in position `i`
            corresponds to the node entity in position `i` of `turl_kwargs['input_ent']`.
            The index in position `i` that has a value of `gnn_i` also points to the position of the node in
            `gnn_kwargs['x']`.
            Thus, `turl_kwargs['input_ent'][0]` and `gnn_kwargs['x'][seed_edge_node_ids[0]]` are two
            embedding representations of the same node.
        """
        # This function takes the same input as TURLGNN.forward()

        # Init node embeddings from the vocabulary and the attention masks
        tok_embeds, ent_embeds, ent_candidates_embeds = self.get_turl_embeds(turl_kwargs)
        attention_tok_mask, attention_ent_mask = self.get_turl_attention_masks(turl_kwargs)
        
        # Initial LM input
        tok_hidden_states, ent_hidden_states = tok_embeds, ent_embeds
        id_ent_mask = turl_kwargs['id_ent_mask']
        # Initial GNN input
        node_feats_x, edge_index, edge_attr = gnn_kwargs['x'], gnn_kwargs['edge_index'], gnn_kwargs['edge_attr']
        x = self.get_gnn_embeds(turl_kwargs, node_ent_ids, node_feats_x)
        x, edge_attr = self.gnn_node_embed(x), self.gnn_edge_embed(edge_attr)

        ## ENCODER
        for (typ, module_types), layer_modules, mix_module in zip(self.layer_types, self.layers, self.mix_modules):
            # Forward pass all layers in parallel
            ## Compute all module embedding updates
            all_module_outputs = {'lm': {'tok': [], 'ent': []}, 'gnn': []}
            for module_typ, module in zip(module_types, layer_modules):
                # LM forward pass
                if module_typ == 'lm':
                    tok_layer_outputs, ent_layer_outputs = module(tok_hidden_states, attention_tok_mask, ent_hidden_states, attention_ent_mask)
                    all_module_outputs['lm']['tok'].append(tok_layer_outputs[0])
                    all_module_outputs['lm']['ent'].append(ent_layer_outputs[0])
                # GNN forward pass
                elif module_typ.startswith('gnn'):
                    x_2 = (x + F.relu(module(x, edge_index, edge_attr))) / 2
                    all_module_outputs['gnn'].append(x_2)
            ## Gather and combine output module embeddings
            # LM pooling
            if len(all_module_outputs['lm']['tok']) > 0:
                tok_hidden_states = self.pooling(all_module_outputs['lm']['tok'])
                ent_hidden_states = self.pooling(all_module_outputs['lm']['ent'])
            # GNN pooling
            if len(all_module_outputs['gnn']) > 0:
                x = self.pooling(all_module_outputs['gnn'])
            ## Seed nodes inter-mixing
            gnn_seed_node_idx, node_ids_where, lm_seed_node_embeds = self.pool_lm_seed_nodes(ent_hidden_states, seed_edge_node_ids, id_ent_mask)
            gnn_seed_node_embeds = x[gnn_seed_node_idx]
            seed_node_embeds = torch.cat((lm_seed_node_embeds, gnn_seed_node_embeds), dim=-1)
            seed_node_embeds = mix_module(seed_node_embeds)
            lm_seed_node_embeds = seed_node_embeds[..., :lm_seed_node_embeds.shape[-1]]
            gnn_seed_node_embeds = seed_node_embeds[..., -gnn_seed_node_embeds.shape[-1]:]
            ## Store new seed node representation (with skip connections)
            # GNN
            x[gnn_seed_node_idx] += gnn_seed_node_embeds
            # LM
            ent_hidden_states[0, id_ent_mask[0]] += lm_seed_node_embeds[node_ids_where]
    
        ## REDOUT HEADS
        turl_outputs = self.get_turl_outputs(tok_hidden_states, ent_hidden_states, ent_candidates_embeds, turl_kwargs)
        gnn_outputs = self.get_gnn_outputs(x, gnn_kwargs)

        return turl_outputs, gnn_outputs

    def gnn_loss_fn(self, input1, input2):
        return -torch.log(input1 + 1e-7).mean() - torch.log(1 - input2 + 1e-7).mean()