"""Module including data parsers to prepare KGDataset data for the different architectures."""
import os
import copy
import logging
import itertools

import pandas as pd
import numpy as np
import torch
import torch_geometric as pyg
from tqdm import tqdm
from munch import Munch

from .data import KGDataset
from .utils import choice
from .turl.config import TURLConfig 

logger = logging.getLogger(__name__)


class Parser:
    """Abstract class for data parsers.
    
    Parameters
    ----------
    dataset : KGDataset
        Dataset that will be parsed. 
    """
    def __init__(self, dataset: KGDataset) -> None:
        self.dataset = dataset
        self.preprocess(dataset)

    # This method may be overwritten by child classes
    # @abstractmethod
    def preprocess(self, dataset: KGDataset):
        """Prepare parser.
        
        This function is called once at initialization.
        """
        pass

    # This method must be overwritten by child classes
    # @abstractmethod
    def parse_df(self, edge_df: pd.DataFrame, node_df: pd.DataFrame):
        """Prepare edge and node data as input to the model."""
        raise NotImplementedError()

    # This method must be overwritten by child classes
    # @abstractmethod
    def collate_fn(self, dataset: KGDataset, samples: list[int], args: Munch, train: bool):
        """Build batch for model training or validation.
        
        This method is called by :class:`~torch.utils.data.DataLoader` during
        training and validation to generate the input batches.

        For example, the simplest content for this method could be as follows::

            edge_df, node_df = dataset.sample_neighs(samples)
            input = self.parse_df(edge_df, node_df)
            return input

        Parameters
        ----------
        dataset : KGDataset
            Train or validation dataset.
        samples : list[int]
            List of edge indicies to pass to :class:`~KGDataset.sample_neighs`.
        args : Munch
            Train and validation arguments from ``train.py``.
        train : bool
            Whether we are training or validating.
        """
        raise NotImplementedError()


class LMParser(Parser):
    """Data parser for TURL.

    .. note::

        Some of this code is adapted from `TURL's official repository <https://github.com/sunlab-osu/TURL/blob/bfec92e942a648695b3910aab42a6f0b679d37fc/data_loader/data_loaders.py>`_.
    
    Parameters
    ----------
    include_node_feats : bool, optional
        Whether to append source and destination node features as extra columns
        in :class:`~LMParser.parse_df`.
    tokenizer : transformers tokenizer
        Tokenizer to use to parse table headers.
    """
    RESERVED_ENT_VOCAB = {0:{'id':'[PAD]', 'count': -1}, 1:{'id':'[ENT_MASK]', 'count': -1}, 2:{'id':'[PG_ENT_MASK]', 'count': -1}, 3:{'id':'[CORE_ENT_MASK]', 'count': -1}}

    def __init__(self, *args, include_node_feats=True, tokenizer=None, **kwargs) -> None:
        self.include_node_feats = include_node_feats
        self.tokenizer = tokenizer
        self.lm_config: TURLConfig = None
        super().__init__(*args, **kwargs)

    def preprocess(self, dataset: KGDataset):
        entity_vocab, id_ent_id_set, non_id_ent_id_set = self._preprocess_entity_vocab(dataset)

        # set attributes
        self.entity_vocab = entity_vocab
        self.id_ent_id_set = id_ent_id_set
        self.non_id_ent_id_set = non_id_ent_id_set
        self.ent2idx = { entity_vocab[x]['id']: x for x in entity_vocab }
        assert len(self.ent2idx) == len(entity_vocab), "entity size mismatch, |ent2idx| = %d |entity_vocab| = %d" % (len(self.ent2idx), len(entity_vocab))
        self.sample_distribution = self._generate_vocab_distribution(entity_vocab)

        self._preprocess_col_to_candidates(dataset)

    def parse_df(self, edge_df: pd.DataFrame, node_df: pd.DataFrame, train: bool, args):
        # Include node features as extra columns to the right
        if self.include_node_feats:
            df_src_node = node_df.loc[edge_df['SRC']]
            df_dst_node = node_df.loc[edge_df['DST']]
            df_src_node = df_src_node.add_prefix('Source ').reset_index(drop=True)
            df_dst_node = df_dst_node.add_prefix('Destination ').reset_index(drop=True)
            df = pd.concat((edge_df.reset_index(drop=True), df_src_node, df_dst_node), axis=1)
        # Remove SRC and DST
        if 'SRC' in df and 'DST' in df:
            del df['SRC'], df['DST']
        # We do not use continuous edge featuers
        for col in self.dataset.config.edge_cont_cols:
            if col in df: del df[col]
        headers = [[i, col] for i, col in enumerate(df.columns)]

        # retrieve entitites
        entity_cells = df.values.astype(np.int32)
        entity_cells = entity_cells.reshape(-1)
        indices = np.einsum('ijk->jki', np.indices(df.shape)).reshape(-1,2)
        # In the original code they get rid of repeating rows, but we won't do it here since we assume
        # our data already does not repeat rows
        # [(i, j), <entity id>]
        entities = [[tuple(pos), ent] for pos, ent in zip(indices, entity_cells)]

        tokenized_headers = [self.tokenizer.encode(z, max_length=args.max_header_length, truncation=True, add_special_tokens=False) for _,z in headers]
        input_tok, input_tok_pos, input_tok_type, tokenized_meta = [], [], [], []
        tokenized_meta_length = len(tokenized_meta)
        input_tok += tokenized_meta
        input_tok_pos += list(range(tokenized_meta_length))
        input_tok_type += [0]*tokenized_meta_length
        tokenized_headers_length = [len(z) for z in tokenized_headers]
        input_tok += list(itertools.chain(*tokenized_headers))
        input_tok_pos += list(itertools.chain(*[list(range(z)) for z in tokenized_headers_length]))
        input_tok_type += [1]*sum(tokenized_headers_length)
        # maps row indices to the entities contained in that row
        start_row_i = entities[0][0][0]

        input_ent = entity_cells.tolist()
        # if column==0, then it is a core ent (3), else it is a normal ent (4)
        input_ent_type = np.where(indices[:, 1] == 0, 3, 4)  
        input_ent_pos = indices.copy()
        input_ent_pos[:, 0] -= start_row_i
        core_entity_mask = (indices[:, 1] == 0).astype(int)

        meta_and_headers_length = tokenized_meta_length+sum(tokenized_headers_length)
        assert len(input_tok) == meta_and_headers_length
        #create input mask
        tok_tok_mask = np.ones([len(input_tok), len(input_tok)], dtype=int)
        if train:
            meta_ent_mask = np.ones([tokenized_meta_length, len(input_ent)], dtype=int)
        else:
            meta_ent_mask = np.zeros([tokenized_meta_length, len(input_ent)], dtype=int)
        header_ent_mask = np.zeros([sum(tokenized_headers_length), len(input_ent)], dtype=int)
        start_i = 0
        header_span = {}
        for h_i, (h_j, _) in enumerate(headers):
            header_span[h_j] = (start_i, start_i+tokenized_headers_length[h_i])
            start_i += tokenized_headers_length[h_i]
        for e_i, (index, _) in enumerate(entities):
            header_ent_mask[header_span[index[1]][0]:header_span[index[1]][1], e_i] = 1
        ent_header_mask = np.transpose(header_ent_mask)
        if not train:
            header_ent_mask = np.zeros([sum(tokenized_headers_length), len(input_ent)], dtype=int)

        input_tok_mask = [tok_tok_mask, np.concatenate([meta_ent_mask, header_ent_mask], axis=0)]
        ent_meta_mask = np.ones([len(input_ent), tokenized_meta_length], dtype=int)

        # CREATE VISIBILITY MASK
        # All entities in the same column see each other
        # All entities in the same row see each other
        same_row = indices[:, 0][:, None] == indices[:, 0][None, :]
        same_col = indices[:, 1][:, None] == indices[:, 1][None, :]
        ent_ent_mask = (same_row | same_col).astype(int)
        input_ent_mask = [np.hstack([ent_meta_mask, ent_header_mask]), ent_ent_mask]

        # Add special [CORE_ENT_MASK], and Page entity (id 0)
        input_ent = [self.ent2idx['[CORE_ENT_MASK]'], 0] + input_ent
        input_ent_type = np.hstack(([3, 2], input_ent_type))
        input_tok_mask[1] = np.hstack([np.zeros([len(input_tok), 2], dtype=int), input_tok_mask[1]])
        input_ent_pos = np.vstack(([[0, 0], [0, 0]], input_ent_pos))

        new_input_ent_mask = [np.ones([len(input_ent), len(input_tok)], dtype=int), np.ones([len(input_ent), len(input_ent)], dtype=int)]
        new_input_ent_mask[0][2:, :] = input_ent_mask[0]
        new_input_ent_mask[1][2:, 2:] = input_ent_mask[1]
        # process [CORE_ENT_MASK] mask
        new_input_ent_mask[0][1, tokenized_meta_length:] = 0
        if 0 in header_span:
            new_input_ent_mask[0][1, tokenized_meta_length+header_span[0][0]:tokenized_meta_length+header_span[0][1]] = 1
        new_input_ent_mask[1][1, 2:] = 0
        new_input_ent_mask[1][2:, 1] = 0
        # process pgEnt mask
        new_input_ent_mask[1][:, 0] = 0
        new_input_ent_mask[1][0, :] = 0
        input_ent_mask = new_input_ent_mask
        core_entity_mask = np.hstack(([0, 1], core_entity_mask))

        # These are the entity IDs for this subtable
        # only include non ID entities
        nonid_ents = self.non_id_ent_id_set.numpy()
        entities_unique = np.unique(entity_cells[np.isin(entity_cells, nonid_ents)]).astype(np.int32)

        ## CANDIDATES
        # entity_cand.shape = (nb_token, nb_candidates (variable))
        entity_cand = self.col_to_candidates[input_ent_pos[:, 1]]
        # For node IDs we instead provide as candidates the IDs in the subgraph
        # Node source and destination entity IDs
        id_col_idx = np.where(np.isin(df.columns, [self.dataset.config.edge_src_col, self.dataset.config.edge_dst_col]))[0]
        id_ents_mask = (input_ent_pos[:, 1] == id_col_idx[0]) 
        for i in range(1, id_col_idx.shape[0]):
            id_ents_mask |= (input_ent_pos[:, 1] == id_col_idx[i])
        for i in np.where(id_ents_mask)[0]:
            entity_cand[i] = np.array([0])

        input_tok_mask = np.hstack(input_tok_mask)
        input_ent_mask = np.hstack(input_ent_mask)

        return [np.array(input_tok),np.array(input_tok_type),np.array(input_tok_pos),input_tok_mask, \
                np.array(input_ent),np.array(input_ent_type),input_ent_pos,input_ent_mask, \
                core_entity_mask,entities_unique,entity_cand]

    def mask_tokens(self, inputs, tokenizer, mlm_probability=0.2):
        """Prepare masked tokens inputs/labels for MLM on the header: 80% MASK, 10% random, 10% original. """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = [list(map(lambda x: 1 if x == tokenizer.pad_token_id else 0, val)) for val in labels.tolist()]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -1  # We only compute loss on masked tokens

        MASK_PROB = 1.0 #0.8
        RAND_WORD_PROB = 0.0 #0.1
        
        RAND_WORD_PROB = 0.0 if RAND_WORD_PROB == 0.0 else RAND_WORD_PROB / (1.0 - MASK_PROB)
        
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, MASK_PROB)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, RAND_WORD_PROB)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def mask_ent(self, config: TURLConfig, is_train, df, edge_ids, inputs_origin, inputs_pos, core_entity_mask, mlm_probability=0.2, 
                 id_ent_id_set=None, ent_mask_prob=0.8, rand_word_prob=0.1,
                 khop=False):
        """Prepare masked entities inputs/labels for masked entity modeling on the cells: 80% MASK, 10% random, 10% original. """
        # ONLY PREDICT ON SEED EDGES
        if khop:
            # Get seed edges in relative row position
            seed_edges = np.where(np.isin(df.index, edge_ids))[0]
            # Get seed entities
            seed_entities_mask = torch.isin(inputs_pos[..., 0], torch.tensor(seed_edges))

        labels = inputs_origin.clone()
        inputs = inputs_origin.clone()

        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        special_tokens_mask = (labels<len(self.RESERVED_ENT_VOCAB))
        # special_tokens_mask[:, 1] = True
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        nonid_ent_mask = ~torch.isin(labels, id_ent_id_set)
        # Do not mask entities which are IDs
        masked_indices &= nonid_ent_mask
        # If we do k-hop sampling
        if khop:
            # Only mask entities in seed edges
            masked_indices &= seed_entities_mask
        # If we are evaluating, do not mask node feature entities
        if config.never_mask_node_feats or (not config.mask_node_feats_in_eval and not is_train):
            node_feat_mask = torch.isin(inputs_pos[..., 1], self.node_feat_cols.to(inputs_pos.device))
            masked_indices &= ~node_feat_mask
        labels[~masked_indices] = -1  # We only compute loss on masked tokens

        assert ent_mask_prob + rand_word_prob <= 1.0, f'{ent_mask_prob} + {rand_word_prob} > 1 (!)'
        if ent_mask_prob == 1.0:
            rand_word_prob = 0.0
        else:
            rand_word_prob = rand_word_prob / (1.0 - ent_mask_prob)
        
        # 80% of the time, we replace masked input ent with [ENT_MASK]/[PG_ENT_MASK]/[CORE_ENT_MASK] accordingly
        pg_ent_mask = torch.zeros(labels.shape)
        pg_ent_mask[:,0] = 1
        indices_replaced = torch.bernoulli(torch.full(labels.shape, ent_mask_prob)).bool() & masked_indices
        inputs[indices_replaced] = self.ent2idx['[ENT_MASK]']
        inputs[indices_replaced & pg_ent_mask.bool()] = self.ent2idx['[PG_ENT_MASK]']
        inputs[indices_replaced & core_entity_mask] = self.ent2idx['[CORE_ENT_MASK]']

        # 10% of the time, we replace masked input entity with random entity from the entire vocabulary
        indices_random = torch.bernoulli(torch.full(labels.shape, rand_word_prob)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(low=len(self.RESERVED_ENT_VOCAB),high=len(self.ent2idx), size=labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        inputs[:, 1] = inputs_origin[:, 1]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def generate_random_candidate(self, args, batch_entities, batch_entities_unique, candidate_distribution=None, batch_ent_cand=None):
        # give each entity a probability in [0, 1] to be picked
        random_shifts = np.random.random(len(self.ent2idx))
        if candidate_distribution is not None:
            # transform `random_shifts` to a distribution
            random_shifts /= np.sum(random_shifts)
            # make entities with high `candidate_distribution` more 
            # likely to be picked
            random_shifts -= candidate_distribution
        all_batch_ents = np.concatenate(batch_entities_unique, axis=0)
        random_shifts[all_batch_ents] = 10
        all_batch_ents = set(all_batch_ents)
        # select `max_entity_candidate` candidates per batch with lowest random
        # value from `random_shifts`
        # shape = (batch_size, self.max_entity_candidate)
        # After this, `final_candidates` contains random entities from the entity vocabulary
        # final_candidates.shape = (batch_size, max_entity_candidate)
        max_nb_ents = max(len(e) for e in batch_ent_cand)
        # shape = (batch_size, nb_tokens, max_nb_candidates)
        final_candidates = np.zeros((batch_entities_unique.shape[0], max_nb_ents, args.max_entity_candidate), dtype=int)

        for batch_i, (ents, unique_ents) in enumerate(zip(batch_entities, batch_entities_unique)):
            E = len(batch_ent_cand[batch_i])
            N = args.max_entity_candidate
            # final_candidates.shape = (batch_size, nb_entities, max_nb_candidates)
            # batch_ent_cand.shape = (batch_size, nb_entities, nb_candidates (variable))
            # rand_idx = np.random.choice(batch_ent_cand.shape[2], N, replace=False)
            for i in range(E):
                M = len(batch_ent_cand[batch_i][i])
                if N < M:
                    # Our choice function without replacement is much faster than numpy's,
                    # although it is still slow
                    # rand_idx = np.random.choice(len(batch_ent_cand[batch_i][i]), min(M, N), replace=False)
                    rand_idx = choice(len(batch_ent_cand[batch_i][i]), N)
                    cands = batch_ent_cand[batch_i][i][rand_idx]
                else:
                    cands = batch_ent_cand[batch_i][i]
                final_candidates[batch_i, i, :M] = cands
                # Add "true" candidate at random position
                true_i = np.random.choice(min(M, N), 1)[0]
                final_candidates[batch_i, i, true_i] = ents[i]
        return final_candidates

    @property
    def edge_feat_cols(self):
        return torch.arange(2, 2+len(self.dataset.config.edge_cat_cols))

    @property
    def node_feat_cols(self):
        E = len(self.dataset.config.edge_cat_cols)
        N = len(self.dataset.config.node_cat_cols)
        return torch.arange(2+E, 2+E+2*N)

    def _preprocess_col_to_candidates(self, dataset: KGDataset):
        edge_cols = [dataset.config.edge_src_col, dataset.config.edge_dst_col] + dataset.config.edge_cat_cols
        node_cols = dataset.config.node_cat_cols
        self.col_to_candidates = [np.array([self.ent2idx['[PAD]']])] * (len(edge_cols) + 2*len(node_cols)) 

        logger.info('Computing per-column candidates ..')

        # Include edge feature columns
        for col_i, col in tqdm(enumerate(edge_cols), total=len(edge_cols), desc='Edge columns'):
            col_data = dataset.edge_data[col].map(lambda x: self.ent2idx.get(x, float('nan')))
            # Fill NaN with [PAD]
            col_data.fillna(self.ent2idx['[PAD]'], inplace=True)
            self.col_to_candidates[col_i] = col_data.unique()
        # Include node feature columns. These columns are concat at the end for SRC and DST nodes
        for i in (0, 1):
            for col_i, col in tqdm(enumerate(node_cols), total=len(node_cols), desc=f'Node {i} columns'):
                col_data = dataset.node_data[col].map(lambda x: self.ent2idx.get(x, float('nan')))
                # Fill NaN with [PAD]
                col_data.fillna(self.ent2idx['[PAD]'], inplace=True)
                self.col_to_candidates[len(edge_cols) + i*len(node_cols) + col_i] = col_data.unique()
        self.col_to_candidates = np.array(self.col_to_candidates, dtype=object)

    def _generate_vocab_distribution(self, entity_vocab):
        """Returns array with the log10 of the frequency of the entity in the dataset."""
        distribution = np.zeros(len(entity_vocab))
        for i, v in entity_vocab.items():
            if i in self.RESERVED_ENT_VOCAB:
                distribution[i] = 2
            else:
                distribution[i] = int(v['count'])
        distribution = np.log10(distribution)
        distribution /= np.sum(distribution)
        return distribution
        
    def _preprocess_entity_vocab(self, dataset: KGDataset):
        entity_vocab = copy.deepcopy(self.RESERVED_ENT_VOCAB)

        logger.info('Preprocessing entity vocabulary..')

        src_ids = dataset.edge_data[dataset.config.edge_src_col].value_counts()
        dst_ids = dataset.edge_data[dataset.config.edge_dst_col].value_counts()
        id_counts = src_ids.add(dst_ids, fill_value=0)

        def _insert_idx_counts(index, counts, prefix='', id_to_ent_id=None):
            for id, count in tqdm(zip(index, counts), total=len(counts)):
                id = prefix + str(id)
                ent_id = len(entity_vocab)
                entity_vocab[ent_id] = { 'id': id, 'count': count }
                if id_to_ent_id is not None:
                    id_to_ent_id[id] = ent_id

        # mapping between holding and asset IDs and their respective entity ID
        id_to_ent_id = dict()

        _insert_idx_counts(id_counts.index, id_counts, id_to_ent_id=id_to_ent_id)

        # extract other categorical data
        cat_counts = None
        for cat_col in dataset.config.edge_cat_cols:
            counts = dataset.edge_data[cat_col].value_counts()
            cat_counts = counts if cat_counts is None else cat_counts.add(counts, fill_value=0)
        for cat_col in dataset.config.node_cat_cols:
            counts = dataset.node_data[cat_col].value_counts()
            cat_counts = counts if cat_counts is None else cat_counts.add(counts, fill_value=0)
        
        _insert_idx_counts(cat_counts.index, cat_counts)

        id_ent_id_set = torch.tensor(list(set(id_to_ent_id.values()))).cpu()
        non_ent_ids = set(entity_vocab.keys()) - set(id_to_ent_id.values())
        non_id_ent_id_set = torch.tensor(list(non_ent_ids)).cpu()
            
        logger.info('total number of entities: %d' % (len(entity_vocab)))
        logger.info(f'# of All IDs in dataset: {len(id_ent_id_set)}')
        return entity_vocab, id_ent_id_set, non_id_ent_id_set

    def _collate_fn(self, dataset: KGDataset, samples, df, batch, args, train):
        edge_ids = dataset.edge_data.index[samples]

        batch = [v[None, ...] if isinstance(v, np.ndarray) else [v] for v in batch]
        input_tok,input_tok_type,input_tok_pos,input_tok_mask,input_ent,input_ent_type,input_ent_pos,input_ent_mask,core_entity_mask,entities_unique,entity_cand = batch

        input_tok = torch.LongTensor(input_tok)
        input_tok_type = torch.LongTensor(input_tok_type)
        input_tok_pos = torch.LongTensor(input_tok_pos)
        input_tok_mask = torch.LongTensor(input_tok_mask)

        input_ent = torch.LongTensor(input_ent)
        input_ent_type = torch.LongTensor(input_ent_type)
        input_ent_pos = torch.LongTensor(input_ent_pos)
        input_ent_mask = torch.LongTensor(input_ent_mask)
        core_entity_mask = torch.BoolTensor(core_entity_mask)
        
        input_tok_final, input_tok_labels = self.mask_tokens(input_tok, self.tokenizer, mlm_probability=args.mlm_probability)
        input_ent_final, input_ent_labels = self.mask_ent(self.lm_config, train, df, edge_ids, input_ent, input_ent_pos, 
                                                          core_entity_mask, 
                                                          mlm_probability=args.ent_mlm_probability,
                                                          id_ent_id_set=self.id_ent_id_set,
                                                          ent_mask_prob=args.ent_mask_prob, rand_word_prob=args.rand_word_prob,
                                                          khop=dataset.num_neighbors is not None)
        
        ent_candidates = self.generate_random_candidate(args, input_ent, entities_unique, self.sample_distribution, batch_ent_cand=entity_cand)
        ent_candidates = torch.LongTensor(ent_candidates)

        # fix position embeddings
        input_tok_pos[:, :] = torch.arange(input_tok_pos.shape[1])

        return {'input_tok': input_tok_final, 'input_tok_type': input_tok_type, 'input_tok_pos': input_tok_pos, 'tok_masked_lm_labels': input_tok_labels, 
                'input_tok_mask': input_tok_mask, 'input_ent': input_ent_final, 'input_ent_type': input_ent_type, 'input_ent_pos': input_ent_pos, 
                'ent_masked_lm_labels': input_ent_labels, 'input_ent_mask': input_ent_mask, 'ent_candidates': ent_candidates }


class GNNParser(Parser):
    """Data parser for GNNs."""
    def preprocess(self, dataset: KGDataset):
        def generate_onehot_dict(df, column):
            unique_values = df[column].dropna().unique()
            return { c: vec.tolist() for c, vec in zip(unique_values, np.eye(len(unique_values))) }
        
        self.edge_onehot_dicts = {col: generate_onehot_dict(dataset.edge_data, col) for col in dataset.config.edge_cat_cols}
        self.node_onehot_dicts = {col: generate_onehot_dict(dataset.node_data, col) for col in dataset.config.node_cat_cols}
        logger.info(f'Generated one-hot vectors for {len(self.edge_onehot_dicts)} edge columns')
        logger.info(f'Generated one-hot vectors for {len(self.node_onehot_dicts)} node columns')

    def parse_df(self, dataset: KGDataset, df_edges, df_nodes, train, args):
        """This function takes the sampled neighborhood as a table of edges and creates a PyG data object that can be used for GNN link prediction."""
        #0. Padding NaNs
        # Identify 'Payment Amount' columns and fill NaN with 0
        df = df_edges.astype({col: 'float64' for col in df_edges.columns if col in dataset.config.edge_cont_cols})
        df[dataset.config.edge_cont_cols] = df[dataset.config.edge_cont_cols].fillna(0)
        df[dataset.config.edge_cat_cols] = df[dataset.config.edge_cat_cols].fillna('None')
        df_nodes[dataset.config.node_cont_cols] = df_nodes[dataset.config.node_cont_cols].fillna(0)
        df_nodes[dataset.config.node_cat_cols] = df_nodes[dataset.config.node_cat_cols].fillna('None')

        # NEW: compute edge attributes (~ x6 speedup)
        edge_attr = []
        for col in dataset.config.edge_cat_cols:
            # Columns with categorical data
            onehot_dict = self.edge_onehot_dicts[col]
            default = [0] * len(list(onehot_dict.values())[0])
            encoded_col = np.stack(df[col].map(lambda x: onehot_dict.get(x, default)).values)
            edge_attr.append(encoded_col)
        for col in dataset.config.edge_cont_cols:
            # Columns with continuous data
            edge_attr.append(df[col].values[:, None])

        edge_attr = np.concatenate(edge_attr, axis=1)

        #2 node features
        #1. 1-hot encode the node features
        node_feats = []
        for col in dataset.config.node_cat_cols:
            # Columns with categorical data
            onehot_dict = self.node_onehot_dicts[col]
            default = [0] * len(list(onehot_dict.values())[0])
            encoded_col = np.stack(df_nodes[col].map(lambda x: onehot_dict.get(x, default)).values)
            node_feats.append(encoded_col)
        for col in dataset.config.node_cont_cols:
            # Columns with continuous data
            node_feats.append(df_nodes[col].values[:, None])

        node_feats = np.concatenate(node_feats, axis=1)
        
        # Global to local node ID mapping
        #map the node ids from df[['SRC', 'DST']] to range (1, n_nodes) such that they lign up with the node and edge features
        n_id_map = {value: index for index, value in enumerate(df_nodes.index)}

        # Applying the mapping to 'SRC' and 'DST'
        df['SRC_mapped'] = df['SRC'].map(n_id_map)
        df['DST_mapped'] = df['DST'].map(n_id_map)

        #3 Creating the PyG objects
        edge_index = torch.LongTensor(df[['SRC_mapped', 'DST_mapped']].to_numpy().T)
        x = torch.tensor(node_feats).float()
        edge_attr = torch.tensor(edge_attr).float()

        return pyg.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr), n_id_map

    def collate_fn(self, dataset: KGDataset, samples: list[int], args: Munch, train=True):
        edge_df, node_df = dataset.sample_neighs(samples)
        gnn_input = self.parse_df(dataset, edge_df, node_df, train=train, args=args)

        (batch, pos_seed_mask, neg_seed_mask), _ = self._collate_fn(dataset, samples, edge_df, gnn_input, args, train=train)

        return batch, pos_seed_mask, neg_seed_mask

    @property
    def n_node_feats(self):
        return sum(len(v) for v in self.node_onehot_dicts.values()) + len(self.dataset.config.node_cont_cols)

    @property
    def n_edge_feats(self):
        return sum(len(v) for v in self.edge_onehot_dicts.values()) + len(self.dataset.config.edge_cont_cols)

    def _collate_fn(self, dataset: KGDataset, samples, df_edges, gnn_input, args, train):
        batch, n_id_map = gnn_input

        # NEW
        #1. add the negative samples
        E = batch.edge_index.shape[1]

        positions = torch.arange(E)
        drop_count = min(200, int(len(positions) * 0.15)) # 15% probability to drop an edge or maximally 200 edges
        if len(positions) > 0 and drop_count > 0:
            drop_idxs = torch.multinomial(torch.full((len(positions),), 1.0), drop_count, replacement=False) #[drop_count, ]
        else:
            drop_idxs = torch.tensor([]).long()
        drop_positions = positions[drop_idxs]

        mask = torch.zeros((E,)).long() #[E, ]
        mask = mask.index_fill_(dim=0, index=drop_positions, value=1).bool() #[E, ]

        input_edge_index = batch.edge_index[:, ~mask]
        input_edge_attr  = batch.edge_attr[~mask]

        pos_edge_index = batch.edge_index[:, mask]
        pos_edge_attr  = batch.edge_attr[mask]
        
        # Initialize an empty list to store negative edges
        neg_edges = []
        neg_edge_attrs = []

        nodeset = set(range(batch.edge_index.max()+1))

        # Iterate over each positive edge
        for i, edge in enumerate(pos_edge_index.t()):
            src, dst = edge[0], edge[1]

            # Chose negative examples in a smart way
            unavail_mask = (batch.edge_index == src).any(dim=0) | (batch.edge_index == dst).any(dim=0)
            unavail_nodes = torch.unique(batch.edge_index[:, unavail_mask])
            unavail_nodes = set(unavail_nodes.tolist())
            avail_nodes = nodeset - unavail_nodes
            avail_nodes = torch.tensor(list(avail_nodes))
            # Finally, emmulate np.random.choice() to chose randomly amongst available nodes
            indices = torch.randperm(len(avail_nodes))[:64]
            neg_nodes = avail_nodes[indices]
            
            # Generate 32 negative edges with the same source but different destinations
            neg_dsts = neg_nodes[:32]  # Selecting 32 random destination nodes for the source
            neg_edges_src = torch.stack([src.repeat(32), neg_dsts], dim=0)
            
            # Generate 32 negative edges with the same destination but different sources
            neg_srcs = neg_nodes[32:]  # Selecting 32 random source nodes for the destination
            neg_edges_dst = torch.stack([neg_srcs, dst.repeat(32)], dim=0)

            # Add these negative edges to the list
            neg_edges.append(neg_edges_src)
            neg_edges.append(neg_edges_dst)

            # Replicate the positive edge attribute for each of the negative edges generated from this edge
            pos_attr = pos_edge_attr[i].unsqueeze(0)  # Get the attribute of the current positive edge
            replicated_attr = pos_attr.repeat(64, 1)  # Replicate it 64 times (for each negative edge)
            neg_edge_attrs.append(replicated_attr)

        # Concatenate all negative edges to form the neg_edge_index
        neg_edge_index = torch.cat(neg_edges, dim=1)
        neg_edge_attr = torch.cat(neg_edge_attrs, dim=0)

        # Update the batch object
        batch.edge_index, batch.edge_attr, batch.pos_edge_index, batch.pos_edge_attr, batch.neg_edge_index, batch.neg_edge_attr = \
            input_edge_index, input_edge_attr, pos_edge_index, pos_edge_attr, neg_edge_index, neg_edge_attr
        batch.pos_y = torch.ones(pos_edge_index.shape[1]  , dtype=torch.int32)
        batch.neg_y = torch.zeros(neg_edge_index.shape[1], dtype=torch.int32)
        pos_seed_mask = None
        neg_seed_mask = None

        #3. mask for the seed edges
        seed_edge_node_ids = df_edges.iloc[:len(samples)][['SRC', 'DST']].values.ravel()

        # Transform to local GNN ids
        seed_edge_node_ids = [n_id_map[s] for s in seed_edge_node_ids]
        seed_edge_node_ids = torch.tensor(seed_edge_node_ids)

        #4. return the necessary stuff
        return (batch, pos_seed_mask, neg_seed_mask), seed_edge_node_ids


class HybridLMGNNParser(Parser):
    """Data parser for hybrid LM+GNN architectures."""
    def __init__(self, *args, tokenizer=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.lm_parser = LMParser(*args, tokenizer=tokenizer, **kwargs)
        self.gnn_parser = GNNParser(*args, **kwargs)

    def collate_fn(self, dataset: KGDataset, samples, args, train):
        edge_df, node_df = dataset.sample_neighs(samples)

        # NOTE: the first edges are **guaranteed** to be the seed edges (!)
        df_turl_edges = edge_df.iloc[:len(samples)]
        df_turl_nodes = node_df.loc[np.unique(df_turl_edges[['SRC', 'DST']].values)]
        df_gnn_edges  = edge_df.copy()
        df_gnn_nodes  = node_df.copy()

        # Make sure TURL columns are in the proper order (!)
        src_dst_cols = [dataset.config.edge_src_col, dataset.config.edge_dst_col]
        df_turl_edges = df_turl_edges[['SRC', 'DST'] + src_dst_cols + dataset.config.edge_cat_cols]
        df_turl_nodes = df_turl_nodes[dataset.config.node_cat_cols]

        # TURL: Pre-process edge and node data, transform all entities to their entity id from
        # the entity vocabulary
        for col in src_dst_cols + dataset.config.edge_cat_cols:
            df_turl_edges[col] = df_turl_edges[col].map(lambda x: self.lm_parser.ent2idx.get(x, float('nan')))
        df_turl_edges.fillna(0, inplace=True)
        for col in dataset.config.node_cat_cols:
            df_turl_nodes[col] = df_turl_nodes[col].map(lambda x: self.lm_parser.ent2idx.get(x, float('nan')))
        df_turl_nodes.fillna(0, inplace=True)
        # Do not use learned entity vocab for node features
        if args.no_node_vocab:
            # introduce dummy ID
            df_turl_edges[src_dst_cols] = self.lm_parser.id_ent_id_set[0].item()

        # GNN: Transform SRC/DST IDs to entity IDs
        for col in src_dst_cols:
            df_gnn_edges[col] = df_gnn_edges[col].map(lambda x: self.lm_parser.ent2idx[x])

        # TURL
        turl_input = self.lm_parser.parse_df(df_turl_edges, df_turl_nodes, train, args)
        turl_input = self.lm_parser._collate_fn(dataset, samples, df_turl_edges, turl_input, args, train=train)

        # GNN
        if args.mask_gnn_edges:
            df_gnn_edges = self._mask_gnn_edges(turl_input, df_gnn_edges)
        gnn_input = self.gnn_parser.parse_df(dataset, df_gnn_edges, df_gnn_nodes, train, args)

        # Figure out the entity ID for the nodes in the GNN input, in the local
        # order of the GNN input
        _, n_id_map = gnn_input
        gnn_unique_node_ids = list(n_id_map.keys())
        node_ids = df_gnn_edges[['SRC', 'DST']].values.ravel()
        ids = df_gnn_edges[[dataset.config.edge_src_col, dataset.config.edge_dst_col]].values.ravel()
        node_to_id = { node_id: id for node_id, id in zip(node_ids, ids) }
        node_ent_ids = [node_to_id[node_id] for node_id in gnn_unique_node_ids]

        node_ent_ids = torch.tensor(node_ent_ids)[None, :]

        gnn_input, seed_edge_node_ids = self.gnn_parser._collate_fn(dataset, samples, df_gnn_edges, gnn_input, args, train)

        return turl_input, gnn_input, (seed_edge_node_ids, node_ent_ids,)

    def _mask_gnn_edges(self, turl_input, df_gnn_edges):
        # return df_gnn_edges
        input_ent_pos, input_ent_labels = turl_input['input_ent_pos'], turl_input['ent_masked_lm_labels']
        masked_ents_pos = input_ent_pos[input_ent_labels != -1].numpy()
        # entity positions ignore the first 2 cols, so we add 2 to the position
        cells_to_change = np.zeros(df_gnn_edges.shape, dtype=bool)
        cells_to_change[masked_ents_pos[:, 0], 2+masked_ents_pos[:, 1]] = True
        df_gnn_edges[cells_to_change] = np.nan
        return df_gnn_edges
   