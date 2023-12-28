import logging
import json
from io import open

from transformers.configuration_utils import PretrainedConfig

# Setup logging
logger = logging.getLogger(__name__)


class TURLConfig(PretrainedConfig):
    """TURL configuration class.

    Parameters
    ----------
    tok_vocab_size : int, optional
        Vocabulary size for table header tokens.
    ent_vocab_size : int, optional
        Vocabulary size for table cell entities.
    hidden_size : int, optional
        Size of the encoder layers and the pooler layer.
    num_hidden_layers : int, optional
        Number of hidden layers in the Transformer encoder.
    num_attention_heads : int, optional
        Number of attention heads for each attention layer in the Transformer 
        encoder.
    intermediate_size : int, optional
        The size of the "intermediate" (i.e., feed-forward) layer in the 
        Transformer encoder.
    hidden_act : str, optional
        The non-linear activation function (function or string) in the encoder 
        and pooler. If string, "gelu", "relu", "swish" and "gelu_new" are 
        supported.
    hidden_dropout_prob : float, optional
        The dropout probabilitiy for all fully connected layers in the 
        embeddings, encoder, and pooler.
    attention_probs_dropout_prob : float, optional
        The dropout ratio for the attention probabilities.
    max_position_embeddings : int, optional
        The maximum sequence length that this model might ever be used with. 
        Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
    type_vocab_size : int, optional
        Number of token and entity types.
    initializer_range : float, optional
        The sttdev of the truncated_normal_initializer for initializing all 
        weight matrices.
    layer_norm_eps : float, optional
        The epsilon used by LayerNorm.
    max_header_length : int, optional
        Maximum number of tokens in table header.
    mask_node_feats_in_eval : bool, optional
        Whether to mask node features during eval. If ``False``, only edge
        features are masked during eval.
    max_entity_candidate : int, optional
        Maximum number of proposed enitity candidates to use on cell filling.
    never_mask_node_feats : bool, optional
        Whether to never mask node features, neither train and eval. This
        flag overwrites ``mask_node_feats_in_eval``.
    config_json_file : str, optional
        Optional file path with configuration file in JSON format. If
        not ``None``, all other arguments are ignored.
    """
    def __init__(self, *, tok_vocab_size=1000, ent_vocab_size=900000, hidden_size=768, num_hidden_layers=4, num_attention_heads=12, intermediate_size=1200, hidden_act="gelu", hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=5, initializer_range=0.02, layer_norm_eps=1e-12, max_header_length=512, mask_node_feats_in_eval=False, max_entity_candidate=400, never_mask_node_feats=False, config_json_file=None, **kwargs):
        super(TURLConfig, self).__init__(**kwargs)
        if config_json_file is not None:
            with open(config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        else:
            self.vocab_size = tok_vocab_size
            self.ent_vocab_size = ent_vocab_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
            self.max_header_length = max_header_length
            self.mask_node_feats_in_eval = mask_node_feats_in_eval
            self.never_mask_node_feats = never_mask_node_feats
            self.max_entity_candidate = max_entity_candidate