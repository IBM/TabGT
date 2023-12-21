# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel, BertLMPredictionHead,
    BertPredictionHeadTransform, BertAttention,
    BertIntermediate, BertOutput
)

from .config import TURLConfig

# Setup logging
logger = logging.getLogger(__name__)


BertLayerNorm = torch.nn.LayerNorm


class TableEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config: TURLConfig):
        super(TableEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.ent_embeddings = nn.Embedding(config.ent_vocab_size, config.hidden_size, padding_idx=0).cpu()
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.ent_row_pos_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.ent_col_pos_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_tok, input_tok_type, input_tok_pos, input_ent = None, input_ent_type = None, ent_candidates = None, input_ent_pos = None):
        # Token embeddings
        input_tok_embeds = self.word_embeddings(input_tok)
        input_tok_pos_embeds = self.position_embeddings(input_tok_pos)
        input_tok_type_embeds = self.type_embeddings(input_tok_type)

        tok_embeddings = input_tok_embeds + input_tok_pos_embeds + input_tok_type_embeds
        tok_embeddings = self.LayerNorm(tok_embeddings)
        tok_embeddings = self.dropout(tok_embeddings)

        # Entity embeddings
        ent_embeddings = None
        if input_ent is not None:
            input_ent_embeds = self.ent_embeddings(input_ent)
        if input_ent_type is not None:
            input_ent_type_embeds = self.type_embeddings(input_ent_type)
            ent_embeddings = input_ent_embeds + input_ent_type_embeds
        if input_ent_pos is not None:
            assert input_ent_pos.min() >= 0 and input_ent_pos.max() < self.ent_row_pos_embeddings.num_embeddings, \
            f'{input_ent_pos.min()=} {input_ent_pos.max()=} {input_ent_pos.shape=}'
            # input_ent_pos.shape = (batch size, # of entities, 2) contains row/column position
            input_ent_pos_embeds = self.ent_row_pos_embeddings(input_ent_pos[..., 0]) + self.ent_col_pos_embeddings(input_ent_pos[..., 1])
            ent_embeddings = ent_embeddings + input_ent_pos_embeds
        
        if ent_embeddings is not None:
            ent_embeddings = self.LayerNorm(ent_embeddings)
            ent_embeddings = self.dropout(ent_embeddings)

        if ent_candidates is not None:
            ent_candidates_embeddings = self.ent_embeddings(ent_candidates)
        else:
            ent_candidates_embeddings = None

        # Output
        return tok_embeddings, ent_embeddings, ent_candidates_embeddings
        

class TableLayerSimple(nn.Module):
    def __init__(self, config):
        super(TableLayerSimple, self).__init__()
        self.output_attentions = config.output_attentions
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, tok_hidden_states=None, tok_attention_mask=None, ent_hidden_states=None, ent_attention_mask=None):
        tok_outputs, ent_outputs = (None, None), (None, None)
        if tok_hidden_states is not None:
            if ent_hidden_states is not None:
                tok_self_attention_outputs = self.attention(tok_hidden_states, 
                                                            encoder_hidden_states=torch.cat([tok_hidden_states, ent_hidden_states], dim=1), 
                                                            encoder_attention_mask=tok_attention_mask,
                                                            output_attentions=self.output_attentions)
            else:
                tok_self_attention_outputs = self.attention(tok_hidden_states, 
                                                            encoder_hidden_states=tok_hidden_states, 
                                                            encoder_attention_mask=tok_attention_mask,
                                                            output_attentions=self.output_attentions)
            tok_attention_output = tok_self_attention_outputs[0]
            tok_outputs = tok_self_attention_outputs[1:]
            tok_intermediate_output = self.intermediate(tok_attention_output)
            tok_layer_output = self.output(tok_intermediate_output, tok_attention_output)
            tok_outputs = (tok_layer_output,) + tok_outputs

        if ent_hidden_states is not None:
            if tok_hidden_states is not None:
                ent_self_attention_outputs = self.attention(ent_hidden_states, 
                                                            encoder_hidden_states=torch.cat([tok_hidden_states, ent_hidden_states], dim=1), 
                                                            encoder_attention_mask=ent_attention_mask,
                                                            output_attentions=self.output_attentions)
            else:
                ent_self_attention_outputs = self.attention(ent_hidden_states, 
                                                            encoder_hidden_states=ent_hidden_states, 
                                                            encoder_attention_mask=ent_attention_mask,
                                                            output_attentions=self.output_attentions)
            ent_attention_output = ent_self_attention_outputs[0]
            ent_outputs = ent_self_attention_outputs[1:]
            ent_intermediate_output = self.intermediate(ent_attention_output)
            ent_layer_output = self.output(ent_intermediate_output, ent_attention_output)
            ent_outputs = (ent_layer_output,) + ent_outputs
        
        return tok_outputs, ent_outputs


class TableEncoderSimple(nn.Module):
    def __init__(self, config):
        super(TableEncoderSimple, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([TableLayerSimple(config) for _ in range(config.num_hidden_layers)])

    def forward(self, tok_hidden_states=None, tok_attention_mask=None, ent_hidden_states=None, ent_attention_mask=None):
        tok_all_hidden_states = ()
        tok_all_attentions = ()
        ent_all_hidden_states = ()
        ent_all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                tok_all_hidden_states = tok_all_hidden_states + (tok_hidden_states,)
                ent_all_hidden_states = ent_all_hidden_states + (ent_hidden_states,)

            tok_layer_outputs, ent_layer_outputs = layer_module(tok_hidden_states, tok_attention_mask, ent_hidden_states, ent_attention_mask)
            tok_hidden_states = tok_layer_outputs[0]
            ent_hidden_states = ent_layer_outputs[0]

            if self.output_attentions:
                tok_all_attentions = tok_all_attentions + (tok_layer_outputs[1],)
                ent_all_attentions = ent_all_attentions + (ent_layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            tok_all_hidden_states = tok_all_hidden_states + (tok_hidden_states,)
            ent_all_hidden_states = ent_all_hidden_states + (ent_hidden_states,)

        tok_outputs = (tok_hidden_states,)
        ent_outputs = (ent_hidden_states,)
        if self.output_hidden_states:
            tok_outputs = tok_outputs + (tok_all_hidden_states,)
            ent_outputs = ent_outputs + (ent_all_hidden_states,)
        if self.output_attentions:
            tok_outputs = tok_outputs + (tok_all_attentions,)
            ent_outputs = ent_outputs + (ent_all_attentions,)

        return tok_outputs, ent_outputs  # last-layer hidden state, (all hidden states), (all attentions)


class TableModel(BertPreTrainedModel):
    def __init__(self, config, is_simple=True):
        super(TableModel, self).__init__(config)
        self.is_simple = is_simple
        self.config = config

        self.embeddings = TableEmbeddings(config)
        if is_simple:
            self.encoder = TableEncoderSimple(config)
        else:
            raise NotImplementedError()

        self.init_weights()

    def load_pretrained(self, checkpoint, is_bert=True):
        self.embeddings.load_pretrained(checkpoint, is_bert=is_bert)
        self.encoder.load_pretrained(checkpoint, is_bert=is_bert)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, word_embedding_matrix, ent_embedding_matrix):
        assert self.embeddings.word_embeddings.weight.shape == word_embedding_matrix.shape
        assert self.embeddings.ent_embeddings.weight.shape == ent_embedding_matrix.shape
        self.embeddings.word_embeddings.weight.data = word_embedding_matrix
        self.embeddings.ent_embeddings.weight.data = ent_embedding_matrix

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_tok = None, input_tok_type = None, input_tok_pos = None, input_tok_mask = None,
                input_ent = None, input_ent_type = None, input_ent_mask = None, ent_candidates = None, input_ent_pos = None):
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_input_tok_mask, extended_input_ent_mask = None, None
        if input_tok_mask is not None:
            extended_input_tok_mask = input_tok_mask[:, None, :, :]
            extended_input_tok_mask = extended_input_tok_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_input_tok_mask = (1.0 - extended_input_tok_mask) * -10000.0
        if input_ent_mask is not None:
            extended_input_ent_mask = input_ent_mask[:, None, :, :]
            extended_input_ent_mask = extended_input_ent_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_input_ent_mask = (1.0 - extended_input_ent_mask) * -10000.0

        tok_embedding_output, ent_embedding_output, ent_candidates_embeddings = \
            self.embeddings(input_tok, input_tok_type, input_tok_pos, 
                            input_ent, input_ent_type, ent_candidates, input_ent_pos) #disgard ent_pos since they are all 0
        tok_encoder_outputs, ent_encoder_outputs = self.encoder(tok_embedding_output, extended_input_tok_mask, ent_embedding_output, extended_input_ent_mask)
        tok_sequence_output = tok_encoder_outputs[0]
        ent_sequence_output = ent_encoder_outputs[0]

        tok_outputs = (tok_sequence_output, ) + tok_encoder_outputs[1:]  # add hidden_states and attentions if they are here
        ent_outputs = (ent_sequence_output, ) + ent_encoder_outputs[1:]
        return tok_outputs, ent_outputs, ent_candidates_embeddings  # sequence_output, (hidden_states), (attentions)


class TableLMSubPredictionHead(nn.Module):
    """
    only make prediction for a subset of candidates
    """
    def __init__(self, config: TURLConfig, output_dim=None, use_bias=True):
        super(TableLMSubPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        if use_bias:
            self.bias = nn.Embedding.from_pretrained(torch.zeros(config.ent_vocab_size, 1), freeze=False)
        else:
            self.bias = None

    def forward(self, hidden_states, candidates, candidates_embeddings, return_hidden=False):
        hidden_states = self.transform(hidden_states)
        # If each entity gets its own set of candidates
        if len(candidates_embeddings.shape) == 4:
            # hidden_states.shape = (batch_size, nb_token, hidden_size)
            # candidates.shape = (batch_size, nb_token, nb_candidates)
            # candidates_embeddings.shape = (batch_size, nb_token, nb_candidates, hidden_size)
            # scores.shape = (batch_size, nb_token, nb_candidates)
            # Add a dimension so we can broadcast
            hidden_states = hidden_states.unsqueeze(2)
            scores = torch.matmul(hidden_states, torch.transpose(candidates_embeddings,2,3))
            scores = scores[:, :, 0, :]
            if self.bias is not None:
                bias = self.bias(candidates)[:, :, :, 0]
                scores += bias
        # If all entities share the same candidates
        else:
            scores = torch.matmul(hidden_states, torch.transpose(candidates_embeddings,1,2))
            if self.bias is not None:
                bias = torch.transpose(self.bias(candidates),1,2)
                scores += bias
        if return_hidden:
            return (scores,hidden_states)
        else:
            return scores


class TableMLMHead(nn.Module):
    def __init__(self, config):
        super(TableMLMHead, self).__init__()
        self.tok_predictions = BertLMPredictionHead(config)
        self.ent_predictions = TableLMSubPredictionHead(config)

    def forward(self, tok_sequence_output, ent_sequence_output, ent_candidates, ent_candidates_embeddings):
        tok_prediction_scores = self.tok_predictions(tok_sequence_output)
        ent_prediction_scores = self.ent_predictions(ent_sequence_output, ent_candidates, ent_candidates_embeddings)
        return tok_prediction_scores, ent_prediction_scores


class TURL(BertPreTrainedModel):
    def __init__(self, config, is_simple=True):
        super(TURL, self).__init__(config)

        self.table = TableModel(config, is_simple)
        self.cls = TableMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.tok_predictions.decoder
    
    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        tok_output_embeddings = self.get_output_embeddings()
        tok_input_embeddings = self.table.get_input_embeddings()
        if tok_output_embeddings is not None:
            self._tie_or_clone_weights(tok_output_embeddings, tok_input_embeddings)

    def load_pretrained(self, checkpoint):
        self.table.load_pretrained(checkpoint)
        self.cls.load_pretrained(checkpoint)

    def forward(self, input_tok, input_tok_type, input_tok_pos, input_tok_mask,
                input_ent, input_ent_type, input_ent_pos, input_ent_mask, ent_candidates,
                tok_masked_lm_labels, ent_masked_lm_labels, exclusive_ent_mask=None, 
                edge_feat_ent_mask=None, node_feat_ent_mask=None, id_ent_mask=None,
                return_ent_hidden_states=False):
        # input_ent_pos is not used (!)
        tok_outputs, ent_outputs, ent_candidates_embeddings = self.table(input_tok, input_tok_type, input_tok_pos, 
                                                                         input_tok_mask, input_ent, input_ent_type, 
                                                                         input_ent_mask, ent_candidates, input_ent_pos)
        
        tok_hidden_states = tok_outputs[0]
        ent_hidden_states = ent_outputs[0]
        tok_prediction_scores, ent_prediction_scores = self.cls(tok_hidden_states, ent_hidden_states, ent_candidates, ent_candidates_embeddings)

        # this is concatenation for the output, it doesn't add values
        tok_outputs = (tok_prediction_scores,) + tok_outputs[1:]  # Add hidden states and attention if they are here
        ent_outputs = (ent_prediction_scores,) + ent_outputs[1:]

        # Header token loss
        if tok_masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            tok_masked_lm_loss = loss_fct(tok_prediction_scores.view(-1, self.config.vocab_size), tok_masked_lm_labels.view(-1))
            tok_outputs = (tok_masked_lm_loss,) + tok_outputs
            
        # Cell entity loss
        if ent_masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            if exclusive_ent_mask is not None:
                # add to ent_prediction_scores -10_000 if the predicted entity is in the same column
                # so that it cannot predict that entity as being the one in the cell
                # (why would you want to do that ??)
                # vec[i][j][index[i][j][k]] += src[i][j][k]
                # for the entities which are supposed to be ignored, set the index to 0 so that
                # it doesn't go out of range
                index = exclusive_ent_mask.clone()
                index[index >= 1000] = 0
                ent_prediction_scores.scatter_add_(dim=2, 
                                                   index=index, 
                                                   src=(1.0 - (exclusive_ent_mask>=1000).float()) * -10000.0)
        
            # `ent_masked_lm_labels` contains the entity IDs as labels.
            # BUT, it should really just be the size of the `max_entity_candidate`,
            # so we have to match the entity IDs from the label with the IDs in the
            # entity candidates
            is_candidate = torch.zeros((*ent_masked_lm_labels.shape, ent_candidates.shape[-1])).bool()
            is_candidate = is_candidate.to(ent_masked_lm_labels.device)
            for batch_i in range(ent_candidates.shape[0]):
                is_candidate[batch_i] = torch.eq(ent_masked_lm_labels[batch_i, :, None], ent_candidates[batch_i])
            labels = is_candidate.int().argmax(dim=-1)

            # OLD: `ent_masked_lm_labels` was in Local entity ID format, which changed from table to table and
            #     did not correspond to the global entity ID from the entity vocabulary. The assumption was
            #     that all tables would contain less than `config.max_entity_candidate` entities, since all
            #     tables were quite small (usually 8 rows).
            #     HOWEVER, `ent_candidates` contained the IDs in the global entity vocab ID format (!)
            # ent_masked_lm_loss = loss_fct(ent_prediction_scores.view(-1, self.config.max_entity_candidate), 
            #                               ent_masked_lm_labels.view(-1))
            # NEW: `ent_masked_lm_labels` contains the entity IDs (from the global entity vocab). 
            #     `labels` contains the id in the entity candidate list
            preds = ent_prediction_scores.view(-1, self.config.max_entity_candidate)
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
            ent_outputs = ((id_ent_loss, edge_ent_loss, node_ent_loss),) + ent_outputs
        if return_ent_hidden_states:
            return tok_outputs, ent_outputs, ent_hidden_states
        else:
            return tok_outputs, ent_outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)