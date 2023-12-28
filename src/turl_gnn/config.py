import logging
import json
import sys
from io import open

from transformers.configuration_utils import PretrainedConfig

# Setup logging
logger = logging.getLogger(__name__)


class TURLGNNConfig(PretrainedConfig):
    """Configuration for TURL + GNN hybrid architecture.

    Parameters
    ----------
    arch_type : str, optional
        The type of hybrid architecture:

        - **parallel**: The two archs are separable.
        - **fused**: Node embeddings are combined at the end of each layer.

    gnn_arch : str, optional
        The type of GNN architecture to use. Currently, only "gine" is 
        supported.
    node_feat_type : str, optional
        Type of node features. Can be:

        - **vocab**: Node features come from the TURL entity vocabulary.
        - **gnn**: Node features are the default GNN input node features.
    
    include_gnn_node_feats : bool, optional
        Whether to include node features to the GNN input by mixing them 
        through a MLP.
    mlp_dropout : float, optional
        Internal dropout on all MLP modules. This includes the input MLP module
        that mixes GNN input node features with node embeddings from the TURL 
        entity vocabulary, and the mixing module in the **fused** architecture.
    lp_loss_w : float, optional
        The link prediction loss coefficient. Use this parameter to scale the
        loss impact during training for link prediction differently than for
        cell filling.
    fused_arch_layers : list[str], optional
        How to distribute the fused layers. ``lm`` means transformer layer,
        while ``gnn`` means GNN layer. Layers can be combined with ``|``:
        
        - ``lm|gnn`` means both layers in parallel.
        - ``lm|lm|gnn`` means two transformer layers and one GNN layer in 
            parallel.

        Each layer can be specified in this format as a list. For example,
        ``["lm", "lm", "lm|gnn", "lm|gnn"]`` defines four layers where the
        first two are single transformer layers and the last two are
        parallel transformer and GNN layers.

        The GNN aggregation is GINe and the tabular transformer is TURL.
    alternate_objective : list[str], optional
        How to alternate the objective during training:

        - If ``cf``, then cell filling (CF) is backprop'd in that step.
        - If ``lp``, then link prediction (LP) is backprop'd in that step.
        - If ``cf+lp``, then both CF and LP are backprop'd in that step.

        For example, ``["cf", "lp", "lp", "cf+lp"]`` would do CF, the next two 
        steps LP, and then both CF and LP. This alternation repeats througout 
        training.
    no_node_vocab : bool, optional
        Whether to not use learned vocab embeddings as input node embeds.
        If set to `True`, you should also set `include_gnn_node_feats` to `True` 
        so as to have input node features.
    config_json_file : str, optional
        Optional file path with configuration file in JSON format. If
        not ``None``, all other arguments are ignored.
    """
    def __init__(self, *, arch_type="parallel", gnn_arch="gine", node_feat_type="vocab", include_gnn_node_feats=True, mlp_dropout=0.0, lp_loss_w=1.0, fused_arch_layers=["lm", "lm", "lm|gnn", "lm|gnn"], alternate_objective=["cf+lp"], no_node_vocab=False, config_json_file=None, **kwargs):
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