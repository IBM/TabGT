import logging
import json
import sys
from io import open

from transformers.configuration_utils import PretrainedConfig

# Setup logging
logger = logging.getLogger(__name__)


class TURLGNNConfig(PretrainedConfig):
    r"""
    Arguments
    ---------
    arch_type : str
        The type of TabTran+GNN arch::

        * `parallel`: the two archs are separable
        * `sequential`: TabTran -> GNN
    node_feat_type : str
        Type of node features for non-seed nodes. Can be::

        * `vocab`: features come from the TURL entity vocabulary
    fused_arch_layers : list[str]
        How to distribute the fused layers. `lm` means transformer layer,
        `gnn` means GNN layer. Layers can be combined:: 
        
        * `lm|gnn` means both layers in parallel. We can also have multiple like `lm|lm|gnn`.

        The GNN aggregation is GINe.
    alternate_objective : list[str]
        How to alternate the objective during training::

        * If `cf`, then cell filling is done in that step
        * If `lp`, then link prediction is done in that step
        * If `cf+lp`, then both cell filling and link prediction are done in that step

        For example, `["cf", "lp", "lp", "cf+lp"]` would do CF, the next two steps LP, and then
        both CF and LP. This alternation repeats througout training.
    no_node_vocab : bool
        Whether to not use learned vocab embeddings as input node embeds.
        If set to `True`, you should also set `include_gnn_node_feats` to `True` so
        as to have input node features.
    """
    def __init__(self, arch_type="parallel", gnn_arch="gine", node_feat_type="vocab", include_gnn_node_feats=True, mlp_dropout=0.0, lp_loss_w=1.0, fused_arch_layers=["lm", "lm", "lm|gnn", "lm|gnn"], alternate_objective=["cf+lp"], no_node_vocab=False, config_json_file=None, **kwargs):
        super(TURLGNNConfig, self).__init__(**kwargs)

        if isinstance(config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(config_json_file, unicode)):
            with open(config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        else:
            self.gnn_arch = gnn_arch
            self.node_feat_type = node_feat_type
            self.arch_type = arch_type
            self.include_gnn_node_feats = include_gnn_node_feats
            self.mlp_dropout = mlp_dropout
            self.lp_loss_w = lp_loss_w
            self.fused_arch_layers = fused_arch_layers
            self.alternate_objective = alternate_objective
            self.no_node_vocab = no_node_vocab