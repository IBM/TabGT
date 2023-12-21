import json
import logging


# Setup logging
logger = logging.getLogger(__name__)


class GNNConfig:
    r"""Configuration for model architecture and training/inference.
    
    Arguments
    ---------
    hidden_size : 
        Size of the encoder layers and the pooler layer
    num_hidden_layers : 
        Number of hidden layers in the Transformer encoder
    num_attention_heads : 
        Number of attention heads for each attention layer in
        the Transformer encoder
    intermediate_size : 
        The size of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder
    hidden_act : 
        The non-linear activation function (function or string) in the encoder and pooler. 
        If string, "gelu", "relu", "swish" and "gelu_new" are supported
    hidden_dropout_prob : 
        The dropout probabilitiy for all fully connected layers in the embeddings, encoder, and pooler
    attention_probs_dropout_prob : 
        The dropout ratio for the attention probabilities
    vocab_size : 
        The vocabulary size of the edge feature tokens
    initializer_range : 
        The sttdev of the truncated_normal_initializer for initializing all weight matrices
    layer_norm_eps : 
        The epsilon used by LayerNorm
    """
    def __init__(self, *, n_hidden=64, n_gnn_layers=2, w_ce1=1.0, w_ce2=1.0, dropout=0.1, final_dropout=0.05, n_heads=4, config_json_file=None, n_node_feats=69, n_edge_feats=420, n_classes=2, deg=2, max_nb_input_edges=200):
        if config_json_file is not None:
            with open(config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
            logger.info(f'Loaded configuration file: {config_json_file}')
        else:
            self.n_hidden=n_hidden
            self.n_gnn_layers=n_gnn_layers
            self.w_ce1=w_ce1
            self.w_ce2=w_ce2
            self.dropout=dropout
            self.final_dropout=final_dropout
            self.n_heads=n_heads
            self.n_node_feats=n_node_feats
            self.n_edge_feats=n_edge_feats
            self.n_classes=n_classes
            self.deg=deg
            self.max_nb_input_edges=max_nb_input_edges