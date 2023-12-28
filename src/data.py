"""Module with custom PyTorch dataset class definition."""
import json
import pickle
import logging
from typing import Optional

import torch
import numpy as np
import pandas as pd
import torch_geometric as pyg
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import itertools

# Setup logging
logger = logging.getLogger(__name__)


class KGDataset(Dataset):
    """General knowledge graph (KG) dataset with k-hop sampling capabilities.

    The ``num_neighbors`` attribute determines how to perform k-hop sampling.
    To activate/disable k-hop sampling, call ``set_khop()``.

    .. note::

        PyTorch Geometric (PyG) is used to sample neighbors. For more 
        information see `the PyG documentation <https://pytorch-geometric.readthedocs.io/en/latest/tutorial/neighbor_loader.html>`_.

    Example usage::

    >>> idx = [0, 1, 2, 3]
    >>> dataset.set_khop(enable=True, num_neighbors=[10, 10])
    >>> edge_df, node_df = dataset.sample_neighs(idx)
    
    Parameters
    ----------
    edge_data_path : str
        Path to where the edge data is saved in CSV format.
    node_data_path : str
        Path to where the node data is saved in CSV format.
    config_path : str
        Path to where the dataset configuration file is saved in JSON format.
    sample_outgoing : bool, optional
        Whether to sample outgoing edges, in addition to the incoming ones. The 
        total number of sampled neighbors is still determined by 
        ``num_neighbors`` and ``num_sampled_edges``.
    num_sampled_edges : int, optional
        Maximum number of edges to sample. If ``None``, all sampled edges are
        kept.
    """
    def __init__(self, edge_data_path: str, node_data_path: str, config_path: str, sample_outgoing: bool = True, num_sampled_edges: Optional[int] = None, _skip_loaddata: bool = False):
        self.edge_data_path = edge_data_path
        self.node_data_path = node_data_path
        self.num_neighbors = None
        self.num_sampled_edges = num_sampled_edges
        self.sample_outgoing = sample_outgoing
        # data normalization scalers
        self.scalers = {}
        self._sampling_data = None
        self._sampler = None
        self._node_ids = None

        self.node_data: pd.DataFrame = None
        self.edge_data: pd.DataFrame = None
        self.config: KGDatasetConfig = None
        if not _skip_loaddata:
            self.config = KGDatasetConfig(config_path)
            self.node_data, self.edge_data = self._load_data()
            self._init(None)

    def copy(self) -> "KGDataset":
        """Copy dataset."""
        dataset = self.__class__(edge_data_path=self.edge_data_path, node_data_path=self.node_data_path, config_path=self.config.config_path, sample_outgoing=self.sample_outgoing, num_sampled_edges=self.num_sampled_edges, _skip_loaddata=True)
        # copy all attributes
        for k, v in self.__dict__.items():
            dataset.__dict__[k] = v.copy() if hasattr(v, 'copy') else v
        # call preprocessing function
        dataset._init(self.num_neighbors)
        return dataset

    def set_khop(self, enable: bool, num_neighbors: Optional[list[int]]):
        """Whether to activate/disable k-hop sampling.
        
        Parameters
        ----------
        enable : bool
            Whether to enable or disable k-hop sampling.
        num_neighbors : list[int], optional
            How many neighbors to sample in the k-hop neighborhood of each node.
            For ex., `[10, 5, 2]` samples 10 one-hop, 5 two-hop, and 2 three-hop neighbors, respectively
        """
        if enable:
            assert num_neighbors is not None, f'Provide k-hop # of neighbor sampling'
            self.num_neighbors = num_neighbors
        else:
            self.num_neighbors = None
        self._init(self.num_neighbors) 

    def normalize(self, scalers=None, save_to=None):
        """Normalize node and edge features.
        
        Parameters
        ----------
        scalers : dict[str, scaler], optional
            If specified, the values in all column names matching the keys in 
            ``scalers`` will be scaled according to the saved scaler at that 
            key.

            For example, if ``scalers["Column A"]`` exists and is a sklearn
            scaler, then all values in "Column A" will be transformed according
            to that scaler.

            This dictionnary is modified to add new fitted scalers for all 
            columns not originally present in ``scalers``.
        save_to : str, optional
            If specified, save ``scalers`` to this path in pickle format.

        Returns
        -------
        dict[str, scaler]
            Scalers dictionnary mapping a column name with the corresponding
            scaler.
        """
        if scalers is None:
            scalers = {}

        scaler_classes = {'none': None, 'standard': StandardScaler, 'log-standard': StandardScaler}

        # Edge features
        for cont_col in self.config.edge_cont_cols:
            nan_mask = self.edge_data[cont_col].isna()
            vals = self.edge_data.loc[~nan_mask, cont_col].values[:, None].astype(float)
            # If empty, we skip
            if vals.size == 0:
                continue
            # Get scaler type
            scaler_type = self.config._config['edge_table']['normalization'].get(cont_col, 'standard')
            # Skip if 'none'
            if scaler_type == 'none':
                continue
            # Apply log transform
            if scaler_type.startswith('log'):
                x_low = vals.min()
                vals = np.log(1e-6 + (vals - x_low))
            # Fit new scaler
            if cont_col not in scalers:
                scaler = scaler_classes[scaler_type]()
                scaler.fit(vals)
                scalers[cont_col] = scaler
                logger.info(f'Created new {scaler_type} scaler for edge feature column: {cont_col}')
            else:
                scaler = scalers[cont_col]
            # Transform edge feature
            scaled_x = scaler.transform(vals)[:, 0]
            # Save
            self.edge_data.loc[~nan_mask, cont_col] = scaled_x

        # Node features
        for cont_col in self.config.node_cont_cols:
            nan_mask = self.node_data[cont_col].isna()
            vals = self.node_data.loc[~nan_mask, cont_col].values[:, None].astype(float)
            # If empty, we skip
            if vals.size == 0:
                continue
            # Get scaler type
            scaler_type = self.config._config['node_table']['normalization'].get(cont_col, 'standard')
            # Skip if 'none'
            if scaler_type == 'none':
                continue
            # Apply log transform
            if scaler_type.startswith('log'):
                x_low = vals.min()
                vals = np.log(1e-6 + (vals - x_low))
            # Fit new scaler
            if cont_col not in scalers:
                scaler = scaler_classes[scaler_type]()
                scaler.fit(vals)
                scalers[cont_col] = scaler
                logger.info(f'Created new {scaler_type} scaler for node feature column: {cont_col}')
            else:
                scaler = scalers[cont_col]
            # Transform node feature
            scaled_x = scaler.transform(vals)[:, 0]
            # Save
            self.node_data.loc[~nan_mask, cont_col] = scaled_x

        self.scalers = scalers

        # Save scalers
        if save_to is not None:
            # `save_to` is the path to a pickle file
            with open(save_to, 'wb') as file:
                pickle.dump(self.scalers, file, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f'Saved scalers to: {save_to}')

        return scalers

    def sample_neighs(self, idx, force_no_khop=False) -> (pd.DataFrame, pd.DataFrame):
        """Do k-hop sampling.
        
        If k-hop sampling, this method **guarantees** that the first 
        ``n_seed_edges`` edges in the resulting table are the seed edges in the
        same order as given by ``idx``.

        Parameters
        ----------
        idx : int | list[int] | array
            Edge indices to use as seed for k-hop sampling. These map directly
            to the rows in :class:`~KGDataset.edge_data`.
        force_no_khop : bool, optional
            Whether to forcefully deactivate k-hop sampling in this function 
            call.

        Returns
        -------
        dataframe, dataframe
            Sampled edge and node data.
        """
        if type(idx) is int:
            idx = [idx]
        if self._sampler is None or force_no_khop:
            # If we don't do k-hop sampling, then `idx` corresponds to the edge indices
            # in self.edge_data
            edge_df = self.edge_data.iloc[idx]
        else:
            # Neighborhood sampling
            # Does not support multi-graphs
            node_ids = np.unique(self.edge_data.iloc[idx][['SRC', 'DST']].values, axis=0)
            in_ = pyg.sampler.EdgeSamplerInput(None, torch.tensor(node_ids[:, 0]), torch.tensor(node_ids[:, 1]))
        
            out = self._sampler.sample_from_edges(in_)
    
            perm = self._sampler.edge_permutation 
            e_id = perm[out.edge] if perm is not None else out.edge

            if self.sample_outgoing:
                e_id = e_id % len(self.edge_data)

            # remove repeated edges
            e_id = e_id.unique()

            # always include seed edges (without repeating)
            seed_edges = torch.tensor(idx)
            e_id = e_id[~torch.isin(e_id, seed_edges)]
            e_id = torch.hstack((seed_edges, e_id))

            # fill remaining rows with random rows from the dataset
            n_sampled_edges = e_id.size(0)
            # remove edges if we sample too many
            if self.num_sampled_edges is not None and n_sampled_edges > self.num_sampled_edges:
                e_id = e_id[:self.num_sampled_edges]
            # This should be the same as: table = self.edge_data.iloc[e_id]
            e_id = self.edge_data.index[e_id.numpy()]
            edge_df = self.edge_data.loc[e_id]
        
        # retrieve node data
        node_df = self.node_data.loc[np.unique(edge_df[['SRC', 'DST']].values.ravel())]
        return edge_df, node_df

    def _load_data(self):
        if self.node_data_path is not None:
            node_data = pd.read_csv(self.node_data_path)
        edge_data = pd.read_csv(self.edge_data_path)

        # only keep relevant columns in specific order
        if self.node_data_path is not None:
            node_data = node_data[[self.config.node_id_col] + self.config.node_cat_cols + self.config.node_cont_cols]
        edge_data = edge_data[[self.config.edge_src_col, self.config.edge_dst_col] + self.config.edge_cat_cols + self.config.edge_cont_cols]

        # retrieve node IDs
        ids, edge_index = np.unique(edge_data[[self.config.edge_src_col, self.config.edge_dst_col]].values.ravel(), return_inverse=True)
        # if there is no node data, create node table with a single feature
        if self.node_data_path is None:
            node_data = pd.DataFrame({self.config.node_id_col: ids, 'Dummy feature': np.ones_like(ids)})
            # add it to config
            self.config._config['node_table']['categorical_columns'].append('Dummy feature')
            logger.info('Added dummy node feature column')
        # keep only node IDs
        ids_set = set(ids)
        existing_ids = ids_set & set(node_data[self.config.node_id_col].values)
        logger.info(f'There are {len(ids_set) - len(existing_ids)} ({(len(ids_set) - len(existing_ids))/len(ids_set)*100:.2g}%) missing node IDs in dataset: {self.edge_data_path} {self.node_data_path}')
        # transform IDs to indices
        cols = edge_data.columns.tolist()
        edge_data.loc[:, ['SRC', 'DST']] = edge_index.reshape(-1, 2)
        # rearrange columns
        edge_data = edge_data[['SRC', 'DST'] + cols]

        # add missing nodes as new rows with NaN values
        node_data.set_index(self.config.node_id_col, inplace=True)
        node_data = pd.concat([node_data, pd.DataFrame(index=list(ids_set-existing_ids), columns=node_data.columns)], axis=0)

        # the index of the node_data dataframe corresponds to the indices in SRC and DST of edge_data
        node_data = node_data.loc[ids]
        node_data.reset_index(inplace=True)

        logger.info(f'Loaded {len(node_data)} nodes and {len(edge_data)} edges')

        return node_data, edge_data

    def _init(self, num_neighbors):
        # prepare some variables for k-hop sampling
        edge_index = self.edge_data[['SRC', 'DST']].values
        if self.sample_outgoing:
            # include reverse edges to sample outgoing edges
            edge_index = np.concatenate((edge_index, self.edge_data[['DST', 'SRC']].values), axis=0)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        
        self._node_ids = torch.unique(edge_index)
        if num_neighbors is not None:
            # Warning: `x` argument has to ALWAYS be an arange, otherwise pyg crashes without error
            self._sampling_data = pyg.data.Data(x=torch.arange(edge_index.max()+1), edge_index=edge_index)
            self._sampler = pyg.sampler.neighbor_sampler.NeighborSampler(self._sampling_data, 
                                                                         num_neighbors=num_neighbors)
        else:
            self._sampling_data = None
            self._sampler = None
    
    def __len__(self):
        return len(self.edge_data)

    def __getitem__(self, i):
        return i


def transaction_split(dataset: KGDataset, eval_size: float = 0.2, shuffle: bool = True, seed=42) -> (KGDataset, KGDataset, KGDataset):
    """Split transaction data into train, eval and test sets according to the timestamp.
    
    Parameters
    ----------
    dataset : KGDataset
        Dataset to split.
    
    Returns
    -------
    KGDataset, KGDataset, KGDataset
        Train, eval, and test sets.
    """
    df_edges = dataset.edge_data.copy()
    max_n_id = df_edges.loc[:, ['SRC', 'DST']].to_numpy().max() + 1
    # Assuming dataset.node_data is a df that contains column 'NodeID' which is an np.arange
    df_nodes = dataset.node_data.copy()
    df_nodes['Feature'] = np.ones(max_n_id)

    df_edges['Timestamp'] = df_edges['Timestamp'] - df_edges['Timestamp'].min()

    timestamps = torch.Tensor(df_edges['Timestamp'].to_numpy())
    n_days = int(timestamps.max() / (3600 * 24) + 1)

    #data splitting
    daily_inds, daily_trans = [], [] #irs = illicit ratios, inds = indices, trans = transactions
    for day in range(n_days):
        l = day * 24 * 3600
        r = (day + 1) * 24 * 3600
        day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
        daily_inds.append(day_inds)
        daily_trans.append(day_inds.shape[0])
    
    split_per = [0.6, 0.2, 0.2]
    daily_totals = np.array(daily_trans)
    d_ts = daily_totals
    I = list(range(len(d_ts)))
    split_scores = dict()
    for i,j in itertools.combinations(I, 2):
        if j >= i:
            split_totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
            split_totals_sum = np.sum(split_totals)
            split_props = [v/split_totals_sum for v in split_totals]
            split_error = [abs(v-t)/t for v,t in zip(split_props, split_per)]
            score = max(split_error) #- (split_totals_sum/total) + 1
            split_scores[(i,j)] = score
        else:
            continue

    i,j = min(split_scores, key=split_scores.get)
    # split contains a list for each split (train, validation and test) and each list contains the days that are part of the respective split
    split = [list(range(i)), list(range(i, j)), list(range(j, len(daily_totals)))]
    logging.info(f'Calculate split: {split}')

    # Now, we seperate the transactions based on their indices in the timestamp array
    split_inds = {k: [] for k in range(3)}
    for i in range(3):
        for day in split[i]:
            split_inds[i].append(daily_inds[day]) # split_inds contains a list for each split (tr,val,te) which contains the indices of each day seperately

    tr_inds = [item for sublist in split_inds[0] for item in sublist]
    val_inds = [item for sublist in split_inds[1] for item in sublist]
    te_inds = [item for sublist in split_inds[2] for item in sublist]

    # Training Data
    train_dataset = dataset.copy()
    train_dataset.edge_data = train_dataset.edge_data.iloc[tr_inds]
    tr_nodes = np.unique(train_dataset.edge_data[['SRC', 'DST']].values)
    train_dataset.node_data = train_dataset.node_data.iloc[tr_nodes]
    train_dataset._init(train_dataset.num_neighbors)

    # Validation Data
    eval_dataset = dataset.copy()
    eval_dataset.edge_data = eval_dataset.edge_data.iloc[val_inds]
    eval_nodes = np.unique(eval_dataset.edge_data[['SRC', 'DST']].values)
    eval_dataset.node_data = eval_dataset.node_data.iloc[eval_nodes]
    eval_dataset._init(eval_dataset.num_neighbors)

    # Testing Data
    test_dataset = dataset.copy()
    test_dataset.edge_data = test_dataset.edge_data.iloc[te_inds]
    test_nodes = np.unique(test_dataset.edge_data[['SRC', 'DST']].values)
    test_dataset.node_data = test_dataset.node_data.iloc[test_nodes]
    test_dataset._init(test_dataset.num_neighbors)

    return train_dataset, eval_dataset, test_dataset


def edge_wise_random_split(dataset: KGDataset, eval_size: float = 0.2, shuffle: bool = True, seed=42) -> (KGDataset, KGDataset):
    """Split dataset into train and eval at random.
    
    For a random split, set ``shuffle`` to ``True``.
    
    Parameters
    ----------
    dataset : KGDataset
        Dataset to split.
    eval_size : float, optional
        Fraction size of the eval set. Must be in ``[0, 1]``.
    shuffle : bool, optional
        Whether to randomly shuffle the edge indices before the split.
    seed : int, optional
        PRNG seed for the random shuffling.
    
    Returns
    -------
    KGDataset, KGDataset
        Train and eval sets.
    """
    # set seed for split
    np.random.seed(seed)

    edge_idx = np.arange(len(dataset.edge_data))
    if shuffle:
        np.random.shuffle(edge_idx)
    split_i = int(len(edge_idx) * (1.0-eval_size))
    train_edges = edge_idx[:split_i]
    eval_edges = edge_idx[split_i:]

    # Train set
    logger.info('Creating trainset..')
    train_dataset = dataset.copy()
    train_dataset.edge_data = train_dataset.edge_data.iloc[train_edges]
    train_nodes = np.unique(train_dataset.edge_data[['SRC', 'DST']].values)
    train_dataset.node_data = train_dataset.node_data.loc[train_nodes]
    train_dataset._init(train_dataset.num_neighbors)

    # Validation set
    logger.info('Creating evalset..')
    eval_dataset = dataset.copy()
    eval_dataset.edge_data = eval_dataset.edge_data.iloc[eval_edges]
    eval_nodes = np.unique(eval_dataset.edge_data[['SRC', 'DST']].values)
    eval_dataset.node_data = eval_dataset.node_data.loc[eval_nodes]
    eval_dataset._init(eval_dataset.num_neighbors)

    assert len(eval_dataset.edge_data) + len(train_dataset.edge_data) == len(dataset.edge_data)

    # Log node overlap across splits
    if hasattr(train_dataset, '_node_ids') and train_dataset._node_ids is not None:
        mask = ~torch.isin(train_dataset._node_ids, eval_dataset._node_ids)
        logger.info(f'Nodes in training set not in validation set: {mask.int().sum()} ({mask.float().mean()*100:.2f}%)')
        mask = ~torch.isin(eval_dataset._node_ids, train_dataset._node_ids)
        logger.info(f'Nodes in validation set not in train set: {mask.int().sum()} ({mask.float().mean()*100:.2f}%)')

    return train_dataset, eval_dataset


class KGDatasetConfig:
    """KGDataset configuration class.

    The configuration is defined by a JSON file that specifies the data type of
    the different table columns in the following format::

        {
            "edge_table": {
                "source_id_column": "From ID",
                "destination_id_column": "To ID",
                "categorical_columns": [ ... ],
                "continuous_columns": [ ... ],
                "normalization": { ... }
            },
            "node_table": {
                "id_column": "Node ID",
                "categorical_columns": [ ... ],
                "continuous_columns": [ ... ],
                "normalization": { ... }
            }
        }

    Where ``edge_table`` refers to the CSV table with the edge data and 
    ``node_table`` is the CSV table with the node data.
    
    You can find an example config file under ``data/config/example.json`` from
    the root project folder.
    
    Parameters
    ----------
    config_path : str
        Path to JSON file including dataset configuration.
    """
    def __init__(self, config_path: str) -> None:
        self._config = None
        self.config_path = config_path
        self._parse_config(config_path)

    @property
    def node_id_col(self):
        return self._config['node_table']['id_column']

    @property
    def node_cat_cols(self):
        return self._config['node_table']['categorical_columns']

    @property
    def node_cont_cols(self):
        return self._config['node_table']['continuous_columns']

    @property
    def edge_src_col(self):
        return self._config['edge_table']['source_id_column']

    @property
    def edge_dst_col(self):
        return self._config['edge_table']['destination_id_column']

    @property
    def edge_cat_cols(self):
        return self._config['edge_table']['categorical_columns']

    @property
    def edge_cont_cols(self):
        return self._config['edge_table']['continuous_columns']

    def _parse_config(self, config_path: str):
        with open(config_path, 'r') as f:
            data = json.load(f)

        # Validate config
        missing_keys = {"edge_table", "node_table"} - set(data.keys())
        assert len(missing_keys) == 0, f'{missing_keys} missing from config file ({config_path})'
        missing_keys = {"source_id_column", "destination_id_column", "categorical_columns", "continuous_columns"} - set(data["edge_table"].keys())
        assert len(missing_keys) == 0, f'{missing_keys} missing from edge_table in config file ({config_path})'
        missing_keys = {"id_column", "categorical_columns", "continuous_columns"} - set(data["node_table"].keys())
        assert len(missing_keys) == 0, f'{missing_keys} missing from node_table in config file ({config_path})'

        self._config = data

    def __getitem__(self, k):
        return self._config[k]
        