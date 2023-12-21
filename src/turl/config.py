import logging
import json
import sys
from io import open

from transformers.configuration_utils import PretrainedConfig

# Setup logging
logger = logging.getLogger(__name__)


class TURLConfig(PretrainedConfig):
    r"""
        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu", "swish" and "gelu_new" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """

    def __init__(self, tok_vocab_size=1000, ent_vocab_size=900000, hidden_size=768, num_hidden_layers=4, num_attention_heads=12, intermediate_size=1200, hidden_act="gelu", hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=5, initializer_range=0.02, layer_norm_eps=1e-12, ids_cold_start=False, max_header_length=512, mask_node_feats_in_eval=False, max_entity_candidate=400, never_mask_node_feats=False, config_json_file=None, **kwargs):
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
            self.ids_cold_start = ids_cold_start
            self.max_header_length = max_header_length
            self.mask_node_feats_in_eval = mask_node_feats_in_eval
            self.never_mask_node_feats = never_mask_node_feats
            self.max_entity_candidate = max_entity_candidate