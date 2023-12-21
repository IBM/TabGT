#!/usr/bin/env python
# coding: utf-8

import logging
import argparse

### ARGUMENTS ###
parser = argparse.ArgumentParser(description="Pretrain TURL+GNN on target dataset")

# General
parser.add_argument("--seed", default=42, type=int, help="PRNG seed")
parser.add_argument("--run_name", default="run", help="Name for the WandB run")
parser.add_argument("--unique_n", default="0", help="Unique number for wandb and output folder")
parser.add_argument("--wandb", action="store_true", help="Whether to enable WandB")
parser.add_argument("--no_save", action="store_true", help="Whether to not save the model weights")
parser.add_argument("--edge_data_path", required=True, help="Path where the edge data is stored")
parser.add_argument("--node_data_path", default=None, help="Path where the node data is stored")
parser.add_argument("--data_config_path", required=True, help="Path where the data configuration file is stored")
parser.add_argument("--log_every", default=500, type=int, help="Every these steps, metrics are logged")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Number of training epochs")

# Architecture
parser.add_argument("--weight_decay", default=0.0, type=float, help="AdamW weight decay")
parser.add_argument("--arch_type", default="fused", help="TURL+GNN architecture to use")
parser.add_argument("--include_gnn_node_feats", action="store_true", help="Whether to include node features in input to GNN node embeddings")
parser.add_argument("--fused_arch_layers", nargs="+", type=str, default=["lm", "lm", "lm|gnn", "lm|gnn"], help="Layers to use if `--arch_type fused`")
parser.add_argument("--no_node_vocab", action="store_true", help="Do not use a learned node vocabulary")

# Data generation
parser.add_argument("--split", default='edge-wise', help="Type of data split")
parser.add_argument("--num_neighbors", nargs="+", type=int, default=[100, 100], help="k-hop sampling # of neighbors")
parser.add_argument("--disable_khop", action="store_true", help="Disable k-hop sampling during training")
parser.add_argument("--candidates", default=400, type=int, help="Max. number of entity candidates")
parser.add_argument("--ent_mlm_probability", default=0.2, type=float, help="Probability to mask an entity (both cell or node ID)")
parser.add_argument("--num_sampled_edges", default=10000000, type=int, help="Max # of edges used as input")

# Training
parser.add_argument("--lr", default=5e-4, type=float, help="Learning rate")
parser.add_argument("--lr_scheduler", default="invsqrt", help="Learning rate scheduler")
parser.add_argument("--batch_size", default=200, type=int, help="Batch size")
parser.add_argument("--ent_mask_prob", default=1.0, type=float, help="Probability to mask an entity instead of replacing with a random word")
parser.add_argument("--rand_word_prob", default=0.0, type=float, help="Probability to replace an entity with a random one from the vocabulary instead of masking it")
parser.add_argument("--edge_feat_w", default=0.5, type=float, help="Loss edge feature weight")
parser.add_argument("--node_feat_w", default=0.5, type=float, help="Loss node feature weight")
parser.add_argument("--load_from", default="", help="Path to TURL+GNN checkpoint to load. Is overwritten by --load_turl_from and --load_gnn_from")
parser.add_argument("--lp_loss_w", default=1.0, type=float, help="Weight for the link prediction loss")
parser.add_argument("--alternate_objective", nargs="+", type=str, default=["cf+lp"], help="How to alternate the objective during training between cell filling (CF) and link prediction (LP)")
parser.add_argument("--scheduler_timescale", default=1000, type=int, help="invsqrt lr scheduler timescale")
parser.add_argument("--disable_mask_gnn_edges", action="store_true", help="Whether to also mask the edge features in the GNN input")

# Validation
parser.add_argument("--mask_node_feats_in_eval", action="store_true", help="Whether to also mask node features during evaluation")

args = parser.parse_args()
MASK_GNN_EDGES = not args.disable_mask_gnn_edges

# Main imports
import torch
import wandb
from munch import munchify
from transformers import AutoTokenizer

from src.turl_gnn.model import TURLGNN, FusedTURLGNN
from src.turl_gnn.config import TURLGNNConfig
from src.turl_gnn.train import train
from src.turl.config import TURLConfig
from src.gnn.config import GNNConfig
from src.utils import set_seed
from src.data import KGDataset, edge_wise_random_split, transaction_split
from src.parsers import HybridLMGNNParser

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)

logging.info(f'Starting with arguments: {args}')

# Set seed
set_seed(seed=args.seed)

dataset = KGDataset(edge_data_path=args.edge_data_path, node_data_path=args.node_data_path, config_path=args.data_config_path)

# Use random_split to create training and validation datasets
# train_dataset, eval_dataset = node_wise_random_split(dataset)
if args.split == 'edge-wise':
    train_dataset, eval_dataset = edge_wise_random_split(dataset, shuffle=True, seed=args.seed)
elif args.split == 'transaction':
    train_dataset, eval_dataset, _ = transaction_split(dataset, shuffle=True, seed=args.seed)
logger.info('Split datasets.')

# Validate
assert not train_dataset.edge_data.index.isin(eval_dataset.edge_data.index).any(), 'Edges from validation are in train!'
assert not eval_dataset.edge_data.index.isin(train_dataset.edge_data.index).any(), 'Edges from train are in validation!'

# Normalize
train_dataset.normalize()
eval_dataset.normalize(scalers=train_dataset.scalers)
logger.info('Normalized training and validation.')

# Load the default BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

parser = HybridLMGNNParser(dataset, tokenizer=tokenizer)

turl_config = TURLConfig(tok_vocab_size=len(tokenizer), ent_vocab_size=len(parser.lm_parser.entity_vocab), mask_node_feats_in_eval=args.mask_node_feats_in_eval, max_entity_candidate=args.candidates, output_attentions=False)
parser.lm_parser.lm_config = turl_config

gnn_config = GNNConfig(n_hidden=128, n_gnn_layers=2, w_ce1=1.0, w_ce2=1.0, dropout=0.1, final_dropout=0.05, n_heads=4, config_json_file=None, n_node_feats=parser.gnn_parser.n_node_feats, n_edge_feats=parser.gnn_parser.n_edge_feats, n_classes=1, deg=2, max_nb_input_edges=1e12)

config = TURLGNNConfig(arch_type=args.arch_type, gnn_arch="gine", include_gnn_node_feats=args.include_gnn_node_feats, lp_loss_w=args.lp_loss_w, fused_arch_layers=args.fused_arch_layers, alternate_objective=args.alternate_objective, no_node_vocab=args.no_node_vocab)

if config.arch_type == "fused":
    model = FusedTURLGNN(config, turl_config, gnn_config)
else:
    model = TURLGNN(config, turl_config, gnn_config)

logger.info(f'Model:\n{model}')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if args.load_from:
    state_dict = torch.load(args.load_from)
    model.load_state_dict(state_dict.pop('model_state_dict'))
    del state_dict
    logger.info(f'Loaded state dict for TURL+GNN from: {args.load_from}')

model.to(device)

# ARGUMENTS AND CONFIG #

train_args = munchify({
    "shuffle_train": True,
    "device": "cuda",
    "save_total_limit": 1,
    "save_epochs": 0 if args.no_save else 1,
    "output_dir": f"./runs/TURL_GNN/{args.unique_n}/{args.run_name}",
    "per_gpu_train_batch_size": args.batch_size, 
    "per_gpu_eval_batch_size": args.batch_size, 
    "disable_khop": args.disable_khop,
    "learning_rate": args.lr,
    "cache_dir": "cache",
    # maximum no. of candidate entities to give the model to choose
    # from. Has to be at least as big as the no. of unique entities
    # in the table (!)
    "max_entity_candidate": turl_config.max_entity_candidate,
    "mlm_probability": 0.2,
    "adam_epsilon": 1e-8,
    "warmup_steps": 0,
    "max_grad_norm": 1.0, 
    "logging_steps": 2_000,
    # Also sends batch info to wandb
    "log_lr_every": args.log_every,
    "eval_log_metrics_every": 500,
    # loss = tok_loss * l + ent_loss * (1 - l)
    "loss_lambda": 0.5,

    "max_header_length": turl_config.max_header_length,
    "mask_gnn_edges": MASK_GNN_EDGES,
})
train_args.update(args.__dict__)

# Training

# Empty CUDA and CPU cache
torch.cuda.empty_cache()
import gc; print(f'Collected {gc.collect()} bytes.')

# Weights & Biases
if args.wandb: wandb.login()

global_step, tr_loss = train(train_args, config, turl_config, gnn_config, train_dataset, model, parser=parser, eval_dataset=eval_dataset, log_wandb=args.wandb, debug=False)