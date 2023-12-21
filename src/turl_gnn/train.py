import glob
import logging
import os
import re
import shutil
from typing import Optional

import torch
import wandb
import pandas as pd
import numpy as np
import sklearn.metrics
from tqdm import trange, tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_inverse_sqrt_schedule, get_constant_schedule

from ..utils import set_seed
from .config import TURLGNNConfig
from .model import TURLGNN
from ..turl.config import TURLConfig
from ..gnn.config import GNNConfig
from ..data import KGDataset
from ..parsers import HybridLMGNNParser

pd.options.mode.chained_assignment = None

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

logger = logging.getLogger(__name__)


def accuracy(output, target, ignore_index=None):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        if ignore_index is None:
            total_valid = float(len(target))
            correct += torch.sum((pred == target).float())
        else:
            total_valid = torch.sum((target != ignore_index).float())
            correct += torch.sum(((pred == target) * (target != ignore_index)).float())
    return correct / total_valid


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def calc_ent_acc(parser: HybridLMGNNParser, ent_prediction_scores, input_ent_labels, ent_candidates, id_ent_id_set, input_ent_pos):
    ignore_mask = input_ent_labels != -1

    ## There are much more entities which are IDs than regular entities 
    id_mask = torch.isin(input_ent_labels, id_ent_id_set)
    nonid_mask = ignore_mask & (~id_mask)
    id_mask = ignore_mask & id_mask
    # Edge and node features
    edge_feat_mask = ignore_mask & torch.isin(input_ent_pos[..., 1], parser.lm_parser.edge_feat_cols)
    node_feat_mask = ignore_mask & torch.isin(input_ent_pos[..., 1], parser.lm_parser.node_feat_cols)
        
    # is_candidate.shape = (batch_size, nb_entities, max_nb_candidates)
    is_candidate = torch.zeros((*input_ent_labels.shape, ent_candidates.shape[-1])).bool()
    is_candidate = is_candidate.to(input_ent_labels.device)
    for batch_i in range(ent_candidates.shape[0]):
        is_candidate[batch_i] = torch.eq(input_ent_labels[batch_i, :, None], ent_candidates[batch_i])
    labels = is_candidate.int().argmax(dim=-1)

    # Validate
    assert is_candidate.any(dim=-1)[ignore_mask].all(), \
     f'There are {(~is_candidate.any(dim=-1)[ignore_mask]).int().sum()} entities not present in the candidate entity list (!): {input_ent_labels[ignore_mask][~is_candidate.any(dim=-1)[ignore_mask]]}'

    y_pred = ent_prediction_scores.argmax(dim=-1)

    ent_acc = (y_pred[ignore_mask] == labels[ignore_mask]).float().mean().item()
    id_ent_acc = (y_pred[id_mask] == labels[id_mask]).float().mean().item()
    nonid_ent_acc = (y_pred[nonid_mask] == labels[nonid_mask]).float().mean().item()
    ent_edgefeat_acc = (y_pred[edge_feat_mask] == labels[edge_feat_mask]).float().mean().item()
    ent_nodefeat_acc = (y_pred[node_feat_mask] == labels[node_feat_mask]).float().mean().item()

    if np.isnan(ent_acc): ent_acc = 0.0
    if np.isnan(id_ent_acc): id_ent_acc = 0.0
    if np.isnan(nonid_ent_acc): nonid_ent_acc = 0.0
    if np.isnan(ent_edgefeat_acc): ent_edgefeat_acc = 0.0
    if np.isnan(ent_nodefeat_acc): ent_nodefeat_acc = 0.0

    return {'ent_acc': ent_acc, 'id_ent_acc': id_ent_acc, 'nonid_ent_acc': nonid_ent_acc, 'edge_feat_acc': ent_edgefeat_acc, 'node_feat_acc': ent_nodefeat_acc}


def compute_mrr(pos_pred, neg_pred, ks):
    pos_pred = pos_pred.detach().clone().cpu().numpy().flatten()
    neg_pred = neg_pred.detach().clone().cpu().numpy().flatten()

    num_positives = len(pos_pred)
    neg_pred_reshaped = neg_pred.reshape(num_positives, 64)

    mrr_scores = []
    keys = [f'hits@{k}' for k in ks]
    hits_dict = {key: 0 for key in keys}
    count = 0

    for pos, neg in zip(pos_pred, neg_pred_reshaped):
        # Combine positive and negative predictions
        combined = np.concatenate([neg, [pos]])  # Add positive prediction to the end

        # Rank predictions (argsort twice gives the ranks)
        ranks = (-combined).argsort().argsort() + 1  # Add 1 because ranks start from 1
        for k, key in zip(ks, keys):
            if ranks[-1] <= k:
                hits_dict[key] += 1
        
        count += 1
        # Reciprocal rank of positive prediction (which is the last one in combined)
        reciprocal_rank = 1 / ranks[-1]
        mrr_scores.append(reciprocal_rank)
    
    for key in keys:
        hits_dict[key] /= count

    # Calculate Mean Reciprocal Rank
    mrr = np.mean(mrr_scores)
    
    return mrr, hits_dict


def lp_compute_metrics(pos_pred, neg_pred, pos_labels, neg_labels):
    # Calculate positive accuracy
    pos_accuracy = np.mean(pos_pred == pos_labels)
    
    # Calculate negative accuracy
    neg_accuracy = np.mean(neg_pred == neg_labels)
    
    # Calculate overall accuracy
    mean_accuracy = (pos_accuracy + neg_accuracy) / 2

    # Calculate TP, FP, FN
    preds = np.concatenate((pos_pred, neg_pred), axis=0)
    labels = np.concatenate((pos_labels, neg_labels), axis=0)
    TP = np.sum((preds == 1) & (labels == 1))
    FP = np.sum((preds == 1) & (labels == 0))
    FN = np.sum((preds == 0) & (labels == 1))

    # Calculate Precision and Recall for positive class
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculate positive F1 Score
    pos_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return { 'lp_mean_acc': mean_accuracy, 'lp_pos_f1': pos_f1 }


def gnn_get_predictions_and_labels(pos_pred, neg_pred, pos_label, neg_label, config) -> dict[str, np.ndarray]:
    pos_pred = (pos_pred >= 0.5).float()
    neg_pred = (neg_pred >= 0.5).float()

    # All the outputs are numpy ndarrays
    return {
        "pos_pred": pos_pred.detach().clone().cpu().numpy().flatten(), "neg_pred": neg_pred.detach().clone().cpu().numpy().flatten(),
        "pos_labels": pos_label.detach().clone().cpu().numpy(), "neg_labels": neg_label.detach().clone().cpu().numpy()
    }


def compute_auc(pos_probs, neg_probs, pos_labels, neg_labels):
    probs = np.concatenate((pos_probs, neg_probs), axis=0)
    labels = np.concatenate((pos_labels, neg_labels), axis=0)

    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, probs, pos_label=1)
    auc = sklearn.metrics.auc(recall, precision)

    return auc


def train(args, config: TURLGNNConfig, turl_config: TURLConfig, gnn_config: GNNConfig, train_dataset: KGDataset, 
          model: TURLGNN, parser: HybridLMGNNParser, eval_dataset: Optional[KGDataset] = None, log_wandb=True, debug=False):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size

    # train dataset is always shuffled
    train_dataset.set_khop(enable=True, num_neighbors=args.num_neighbors)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=args.shuffle_train,
                              collate_fn=lambda samples: parser.collate_fn(train_dataset, samples, args=args, train=True))

    t_total = len(train_loader) * args.num_train_epochs

    args.save_steps = int(len(train_loader) * args.save_epochs)
    logger.info(f"Saving model every {args.save_steps} steps")

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.lr_scheduler == 'invsqrt':
        scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=args.warmup_steps, timescale=args.scheduler_timescale)
    elif args.lr_scheduler == 'const':
        scheduler = get_constant_schedule(optimizer)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d" % len(train_dataset))
    logger.info("  Num Epochs = %d" % args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d" % args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    # WandB
    if log_wandb:
        run = wandb.init(project=f"TURL_GNN-{args.unique_n}", name=args.run_name, config=args)

    global_step = 0
    tr_loss = 0.0
    tok_tr_loss, ent_tr_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=True)
    set_seed(args.seed)  # Added here for reproducibility 
    model.train()
    for epoch_i in train_iterator:
        logger.info(f'****** EPOCH {epoch_i} ******')
        batch_metrics = { 'lr': [], 'loss': [],
            'tok_acc': [], 'ent_acc': [], 'id_ent_acc': [], 'nonid_ent_acc': [], 'edge_feat_acc': [], 'node_feat_acc': [], 
            'lp_mean_acc': [], 'lp_pos_f1': [], 'lp_auc': [], 'lp_mrr': [], 'lp_hits@1': [], 'lp_hits@2': [], 'lp_hits@5': [], 'lp_hits@10': []}

        epoch_preds_labels = {"pos_pred": [], "neg_pred": [], "pos_labels": [], "neg_labels": []}

        alternating_objective_i = 0

        epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch_i}", position=0, disable=False)
        for step, batch in enumerate(epoch_iterator):
            # DELETEME
            if debug and step > 10: break

            # GENERAL
            seed_edge_node_ids, node_ent_ids = batch[2]            
            seed_edge_node_ids, node_ent_ids = seed_edge_node_ids.to(args.device), node_ent_ids.to(args.device)

            # TURL
            turl_kwargs = batch[0]
            turl_kwargs = {k: v.to(args.device) for k, v in turl_kwargs.items()}

            input_tok_labels = turl_kwargs['tok_masked_lm_labels']
            input_ent_labels = turl_kwargs['ent_masked_lm_labels']
            ent_candidates = turl_kwargs['ent_candidates']

            input_ent, input_ent_pos = turl_kwargs['input_ent'], turl_kwargs['input_ent_pos']
            id_ent_mask = torch.isin(input_ent, parser.lm_parser.id_ent_id_set.to(args.device)).detach()
            edge_feat_ent_mask = torch.isin(input_ent_pos[..., 1], parser.lm_parser.edge_feat_cols.to(args.device)).detach()
            node_feat_ent_mask = torch.isin(input_ent_pos[..., 1], parser.lm_parser.node_feat_cols.to(args.device)).detach()
            turl_kwargs.update(dict(edge_feat_ent_mask=edge_feat_ent_mask, node_feat_ent_mask=node_feat_ent_mask, id_ent_mask=id_ent_mask))

            # GNN
            gnn_data, pos_seed_mask, neg_seed_mask = batch[1] 
            gnn_data.to(args.device)
            gnn_kwargs = dict(x=gnn_data.x, edge_index=gnn_data.edge_index, edge_attr=gnn_data.edge_attr, 
                              pos_edge_index=gnn_data.pos_edge_index, pos_edge_attr=gnn_data.pos_edge_attr,
                              neg_edge_index=gnn_data.neg_edge_index, neg_edge_attr=gnn_data.neg_edge_attr)
                    
            # FORWARD PASS
            optimizer.zero_grad()
        
            turl_outputs, gnn_outputs = model(turl_kwargs, gnn_kwargs, seed_edge_node_ids, node_ent_ids)
            del turl_kwargs, gnn_kwargs

            # TURL OUTPUTS
            tok_outputs, ent_outputs = turl_outputs
            tok_loss = tok_outputs[0]  # model outputs are always tuple in transformers (see doc)
            id_ent_loss, edge_ent_loss, node_ent_loss = ent_outputs[0]

            tok_prediction_scores = tok_outputs[1]
            ent_prediction_scores = ent_outputs[1]

            # GNN OUTPUTS
            pos_labels = gnn_data.pos_y
            pos_pred = gnn_outputs[0]
            neg_labels = gnn_data.neg_y
            neg_pred = gnn_outputs[1]
            gnn_loss = model.gnn_loss_fn(pos_pred, neg_pred)
            
            # Compute loss
            ent_loss = (id_ent_loss + edge_ent_loss * args.edge_feat_w + node_ent_loss * args.node_feat_w) / (1 + args.edge_feat_w + args.node_feat_w)
            turl_loss = tok_loss * args.loss_lambda + ent_loss * (1 - args.loss_lambda)

            # We support alternating the objective 
            cf_w, lp_w = {"cf": (1, 0), "lp": (0, 1), "cf+lp": (1, 1)}[config.alternate_objective[alternating_objective_i]]
            alternating_objective_i = (alternating_objective_i + 1) % len(config.alternate_objective)
            loss = turl_loss * cf_w + gnn_loss * config.lp_loss_w * lp_w

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f'{step}, {global_step} - Loss is {loss} (!). \n{turl_loss=} \n{gnn_loss=} \n{tok_loss=} \n{ent_loss=} \n{id_ent_loss=} \n{edge_ent_loss=} \n{node_ent_loss=}')

            # Free some GPU memory
            ent_prediction_scores, input_ent_labels, ent_candidates, input_ent_pos = ent_prediction_scores.detach().cpu(), input_ent_labels.detach().cpu(), ent_candidates.detach().cpu(), input_ent_pos.detach().cpu()
            pos_pred, neg_pred, pos_labels, neg_labels = pos_pred.detach().cpu(), neg_pred.detach().cpu(), pos_labels.detach().cpu(), neg_labels.detach().cpu()
            tok_prediction_scores, input_tok_labels = tok_prediction_scores.detach().cpu(), input_tok_labels.detach().cpu()
            tok_loss, ent_loss = tok_loss.detach().cpu(), ent_loss.detach().cpu()

            # Compute gradient
            loss.backward()

            loss = loss.detach().cpu()

            tr_loss += loss.detach().item()
            tok_tr_loss += tok_loss.detach().item()
            ent_tr_loss += ent_loss.detach().item()
            
            batch_metrics['loss'].append(loss.detach().item())

            # TURL PREDICTIONS
            # save predictions for later per-epoch accuracy
            turl_metrics = calc_ent_acc(parser, ent_prediction_scores, input_ent_labels, ent_candidates, parser.lm_parser.id_ent_id_set, input_ent_pos)
            tok_acc = accuracy(tok_prediction_scores.view(-1, turl_config.vocab_size), input_tok_labels.view(-1), ignore_index=-1).item()
            for k, v in turl_metrics.items():
                batch_metrics[k].append(v)
            batch_metrics['tok_acc'].append(tok_acc)

            # GNN PREDICTIONS
            gnn_mrr, gnn_hits_dict = compute_mrr(pos_pred, neg_pred, [1,2,5,10])
            gnn_auc = compute_auc(pos_pred, neg_pred, pos_labels, neg_labels)
            for key in gnn_hits_dict:
                batch_metrics['lp_'+key].append(gnn_hits_dict[key])
            batch_metrics['lp_mrr'].append(gnn_mrr)
            batch_metrics['lp_auc'].append(gnn_auc)
            gnn_preds_labels = gnn_get_predictions_and_labels(pos_pred, neg_pred, pos_labels, neg_labels, gnn_config)
            lp_metrics = lp_compute_metrics(**gnn_preds_labels)
            for key in lp_metrics:
                batch_metrics[key].append(lp_metrics[key])
            for k, v in gnn_preds_labels.items(): 
                epoch_preds_labels[k].extend(v.tolist())

            ## LOG
            if (step + 1) % args.log_lr_every == 0:
                # Get the current learning rate from the scheduler
                learning_rate = scheduler.get_last_lr()[0]  # Assuming there is only one learning rate
                batch_metrics['lr'].append(learning_rate)
                logger.info(f'\nTrain ' + '| '.join([f'{k}: {v[-1]:.4g}' for k,v in batch_metrics.items()]))

                if log_wandb:
                    wandb.log({f'batch_{k}': v[-1] for k, v in batch_metrics.items()}, step=global_step)
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()  # Update learning rate schedule

            global_step += 1

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, '{}-{}'.format('checkpoint', global_step))
                os.makedirs(output_dir, exist_ok=True)
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training

                torch.save({ 'model_state_dict': model_to_save.state_dict(), 'optimizer_state_dict': optimizer.state_dict() }, os.path.join(output_dir, 'state.tar'))
                torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                logger.info("Saving model checkpoint to %s" % output_dir)

                _rotate_checkpoints(args, 'checkpoint')

        # After epoch ends:
        epoch_metrics = {f'tr_{met}': np.mean(batch_metrics[met]) for met in ['loss', 'tok_acc', 'ent_acc', 'id_ent_acc', 'nonid_ent_acc', 'edge_feat_acc', 'node_feat_acc', 'lp_mean_acc', 'lp_auc', 'lp_mrr', 'lp_hits@1', 'lp_hits@2', 'lp_hits@5', 'lp_hits@10']}
        epoch_preds_labels = {k: np.array(v) for k, v in epoch_preds_labels.items()}
        lp_metrics = lp_compute_metrics(**gnn_preds_labels)
        for k, v in lp_metrics.items():
            epoch_metrics['tr_' + k] = v
        logger.info("***** Train results - EPOCH {} *****".format(epoch_i))
        for k, v in epoch_metrics.items():
            logger.info(f" {k}: {v:.4g}")

        if log_wandb:
            wandb.log(epoch_metrics, step=global_step)
        
        # Log validation metrics
        # clear CUDA cache before evaluating
        torch.cuda.empty_cache()
        results = evaluate(args, parser, config, turl_config, gnn_config, eval_dataset, model, prefix=f'EPOCH {epoch_i}', debug=debug)
        model.train()

        if log_wandb: wandb.log(results, step=global_step)


def evaluate(args, parser: HybridLMGNNParser, config: TURLGNNConfig, turl_config: TURLConfig, gnn_config: GNNConfig, eval_dataset: KGDataset, model, prefix="", debug=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    args.eval_batch_size = args.per_gpu_eval_batch_size
    # Note that DistributedSampler samples randomly
    eval_dataset.set_khop(enable=True, num_neighbors=args.num_neighbors)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                              collate_fn=lambda samples: parser.collate_fn(eval_dataset, samples, args=args, train=False))

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_preds_labels = {"pos_pred": [], "neg_pred": [], "pos_labels": [], "neg_labels": []}
    batch_metrics = { 'loss': [], 'tok_acc': [], 'ent_acc': [], 'id_ent_acc': [], 'nonid_ent_acc': [], 'edge_feat_acc': [], 'node_feat_acc': [], 
        'lp_mean_acc': [], 'lp_pos_f1': [], 'lp_auc': [], 'lp_mrr': [], 'lp_hits@1': [], 'lp_hits@2': [], 'lp_hits@5': [], 'lp_hits@10': []}
    model.eval()
    with torch.no_grad():
        step=0

        for batch in tqdm(eval_loader, desc=f"Evaluating", position=0, leave=False):
            # DELETEME
            step+=1
            if debug and step>10: break

            # GENERAL
            seed_edge_node_ids, node_ent_ids = batch[2]            
            seed_edge_node_ids, node_ent_ids = seed_edge_node_ids.to(args.device), node_ent_ids.to(args.device)

            # TURL
            turl_kwargs = batch[0]
            turl_kwargs = {k: v.to(args.device) for k, v in turl_kwargs.items()}

            input_tok_labels = turl_kwargs['tok_masked_lm_labels']
            input_ent_labels = turl_kwargs['ent_masked_lm_labels']
            ent_candidates = turl_kwargs['ent_candidates']

            input_ent, input_ent_pos = turl_kwargs['input_ent'], turl_kwargs['input_ent_pos']
            id_ent_mask = torch.isin(input_ent, parser.lm_parser.id_ent_id_set.to(args.device)).detach()
            edge_feat_ent_mask = torch.isin(input_ent_pos[..., 1], parser.lm_parser.edge_feat_cols.to(args.device)).detach()
            node_feat_ent_mask = torch.isin(input_ent_pos[..., 1], parser.lm_parser.node_feat_cols.to(args.device)).detach()
            turl_kwargs.update(dict(edge_feat_ent_mask=edge_feat_ent_mask, node_feat_ent_mask=node_feat_ent_mask, id_ent_mask=id_ent_mask))

            # GNN
            gnn_data, pos_seed_mask, neg_seed_mask = batch[1] 
            gnn_data.to(args.device)
            gnn_kwargs = dict(x=gnn_data.x, edge_index=gnn_data.edge_index, edge_attr=gnn_data.edge_attr, 
                            pos_edge_index=gnn_data.pos_edge_index, pos_edge_attr=gnn_data.pos_edge_attr,
                            neg_edge_index=gnn_data.neg_edge_index, neg_edge_attr=gnn_data.neg_edge_attr)
                    
            # FORWARD PASS
            model.train()
            turl_outputs, gnn_outputs = model(turl_kwargs, gnn_kwargs, seed_edge_node_ids, node_ent_ids)
            del turl_kwargs, gnn_kwargs
            torch.cuda.empty_cache()

            tok_outputs, ent_outputs = turl_outputs
            tok_loss = tok_outputs[0]  # model outputs are always tuple in transformers (see doc)
            id_ent_loss, edge_ent_loss, node_ent_loss = ent_outputs[0]

            tok_prediction_scores = tok_outputs[1]
            ent_prediction_scores = ent_outputs[1]

            ent_loss = id_ent_loss + edge_ent_loss * args.edge_feat_w + node_ent_loss * args.node_feat_w
            turl_loss = tok_loss * args.loss_lambda + ent_loss * (1 - args.loss_lambda)

            # GNN OUTPUTS
            pos_labels = gnn_data.pos_y
            pos_pred = gnn_outputs[0]
            neg_labels = gnn_data.neg_y
            neg_pred = gnn_outputs[1]
            gnn_loss = model.gnn_loss_fn(pos_pred, neg_pred)

            # Free some GPU memory
            ent_prediction_scores, input_ent_labels, ent_candidates, input_ent_pos = ent_prediction_scores.detach().cpu(), input_ent_labels.detach().cpu(), ent_candidates.detach().cpu(), input_ent_pos.detach().cpu()
            pos_pred, neg_pred, pos_labels, neg_labels = pos_pred.detach().cpu(), neg_pred.detach().cpu(), pos_labels.detach().cpu(), neg_labels.detach().cpu()
            tok_prediction_scores, input_tok_labels = tok_prediction_scores.detach().cpu(), input_tok_labels.detach().cpu()

            # COMPUTE LOSS
            loss = turl_loss + gnn_loss * config.lp_loss_w

            batch_metrics['loss'].append(loss.mean().item())

            # TURL PREDICTIONS
            # save predictions for later per-epoch accuracy
            turl_metrics = calc_ent_acc(parser, ent_prediction_scores, input_ent_labels, ent_candidates, parser.lm_parser.id_ent_id_set, input_ent_pos)
            tok_acc = accuracy(tok_prediction_scores.view(-1, turl_config.vocab_size), input_tok_labels.view(-1), ignore_index=-1).item()
            for k, v in turl_metrics.items():
                batch_metrics[k].append(v)
            batch_metrics['tok_acc'].append(tok_acc)

            # GNN PREDICTIONS
            gnn_mrr, gnn_hits_dict = compute_mrr(pos_pred, neg_pred, [1,2,5,10])
            gnn_auc = compute_auc(pos_pred, neg_pred, pos_labels, neg_labels)
            for key in gnn_hits_dict:
                batch_metrics['lp_'+key].append(gnn_hits_dict[key])
            batch_metrics['lp_mrr'].append(gnn_mrr)
            batch_metrics['lp_auc'].append(gnn_auc)
            gnn_preds_labels = gnn_get_predictions_and_labels(pos_pred, neg_pred, pos_labels, neg_labels, gnn_config)
            lp_metrics = lp_compute_metrics(**gnn_preds_labels)
            for key in lp_metrics:
                batch_metrics[key].append(lp_metrics[key])
            for k, v in gnn_preds_labels.items(): 
                eval_preds_labels[k].extend(v.tolist())

            if step % args.eval_log_metrics_every == 0:
                logger.info(f'\nEval ' + '| '.join([f'{k}: {v[-1]:.4g}' for k,v in batch_metrics.items()]))


    eval_metrics = {f'ev_{met}': np.mean(batch_metrics[met]) for met in ['loss', 'tok_acc', 'ent_acc', 'id_ent_acc', 'nonid_ent_acc', 'edge_feat_acc', 'node_feat_acc', 'lp_mean_acc', 'lp_auc', 'lp_mrr', 'lp_hits@1', 'lp_hits@2', 'lp_hits@5', 'lp_hits@10']}
    eval_preds_labels = {k: np.array(v) for k, v in eval_preds_labels.items()}
    lp_metrics = lp_compute_metrics(**gnn_preds_labels)
    for k, v in lp_metrics.items():
        eval_metrics['ev_' + k] = v

    result = eval_metrics

    logger.info("***** {} - Eval results *****".format(prefix))
    for key in sorted(result.keys()):
        logger.info("  %s = %s" % (key, str(result[key])))

    return result

