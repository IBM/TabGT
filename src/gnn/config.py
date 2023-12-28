import json
import logging

# Setup logging
logger = logging.getLogger(__name__)


class GNNConfig:
    """Configuration for GNN model architecture.
    
    Parameters
    ----------
    n_hidden : int, optional
        Size of hidden node embeddings.
    n_gnn_layers : int, optional
        Number of hidden layers in the GNN encoder.
    num_attention_heads : int, optional
        Number of attention heads for each attention layer in
        the Transformer encoder
    final_dropout : float, optional
        Dropout in readout MLP layer.
    n_node_feats : int, optional
        Number of input node features.
    n_edge_feats : int, optional
        Number of input edge features.
    n_classes : int, optional
        Number of output classes.
    deg : int | tensor, optional
        Histogram of in-degrees of nodes in the training set, used by PNA.
    config_json_file : str, optional
        Optional file path with configuration file in JSON format. If
        not ``None``, all other arguments are ignored.
    """
    def __init__(self, *, n_hidden=64, n_gnn_layers=2, final_dropout=0.05, n_node_feats=69, n_edge_feats=420, n_classes=2, deg=2, config_json_file=None):
        if config_json_file is not None:
            with open(config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
            logger.info(f'Loaded configuration file: {config_json_file}')
        else:
            self.n_hidden=n_hidden
            self.n_gnn_layers=n_gnn_layers
            self.final_dropout=final_dropout
            self.n_node_feats=n_node_feats
            self.n_edge_feats=n_edge_feats
            self.n_classes=n_classes
            self.deg=deg