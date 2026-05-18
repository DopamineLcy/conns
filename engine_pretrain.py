# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
import os
import csv
from typing import Iterable
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import AutoImageProcessor, AutoTokenizer

import utils.misc as misc
import utils.lr_sched as lr_sched
import numpy as np


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    dataset=None):

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', misc.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if data_iter_step % accum_iter == 0:
            current_iter = epoch * len(data_loader) + data_iter_step
            lr_sched.adjust_learning_rate(optimizer, current_iter, args)
        
        if 'debug' in args.note:
            # Accumulate sampling counts from the batch.
            if misc.is_main_process() and dataset is not None and hasattr(dataset, 'entity_sampling_counts'):
                if '_sampling_counts' in batch:
                    for entity_name, count in batch['_sampling_counts'].items():
                        dataset.entity_sampling_counts[entity_name] += count
                    dataset.total_batches_processed += 1
            
            # Save entity sampling histograms on the main process.
            if data_iter_step % 10 == 0 and misc.is_main_process():
                current_dataset = dataset
                if current_dataset is None and hasattr(data_loader, 'dataset'):
                    current_dataset = data_loader.dataset
                
                if current_dataset is not None and hasattr(current_dataset, 'plot_entity_sampling_histogram'):
                    if args and hasattr(args, 'output_dir') and args.output_dir:
                        save_path = os.path.join(args.output_dir, 
                                                f"entity_sampling_histogram_epoch_{epoch}_iter_{data_iter_step}.png")
                        current_dataset.plot_entity_sampling_histogram(save_path=save_path)
                        
                        # Print periodic sampling statistics.
                        if data_iter_step % 100 == 0:
                            stats = current_dataset.get_sampling_statistics()
                            print(f"\n[Entity Sampling Stats at Epoch {epoch}, Iter {data_iter_step}]")
                            print(f"  Total entities: {stats['total_entities']}")
                            print(f"  Sampled entities: {stats['sampled_entities']}")
                            print(f"  Total samples: {stats['total_samples']}")
                            print(f"  Avg samples per entity: {stats['avg_samples_per_entity']:.2f}")
                            entity_counts = stats.get('entity_counts', {})
                            if entity_counts:
                                max_entity = max(entity_counts.items(), key=lambda x: x[1])[0]
                                print(f"  Max samples: {stats['max_samples']} (entity: {max_entity})")
                            else:
                                print(f"  Max samples: {stats['max_samples']}")
                            print(f"  Min samples: {stats['min_samples']}")
                            print(f"  Total batches processed: {stats['total_batches']}\n")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(batch, device=device)
            loss = outputs["loss"]
            if "loss_contrastive" in outputs:
                loss_contrastive = outputs["loss_contrastive"]
            else:
                loss_contrastive = torch.tensor(0.0, device=device)
            # Support both loss_router (V5) and loss_ortho (V6)
            loss_router = outputs.get("loss_router", outputs.get("loss_ortho", torch.tensor(0.0, device=device)))

        loss_value = loss.item()
        loss_contrastive_value = loss_contrastive.item()
        loss_router_value = loss_router.item() if isinstance(loss_router, torch.Tensor) else loss_router
        logit_scale = outputs.get("logit_scale", torch.tensor(0.0)).item()
        logit_bias = outputs.get("logit_bias", torch.tensor(0.0)).item()
        # Handle cases where attn_temperature_exp or others might be None or 0.0
        attn_temperature_exp = outputs.get("attn_temperature_exp", torch.tensor(0.0))
        if isinstance(attn_temperature_exp, torch.Tensor): attn_temperature_exp = attn_temperature_exp.item()
        
        loss_temperature_exp = outputs.get("loss_temperature_exp", torch.tensor(0.0))
        if isinstance(loss_temperature_exp, torch.Tensor): loss_temperature_exp = loss_temperature_exp.item()

        neg_temperature_exp = outputs.get("neg_temperature_exp", torch.tensor(0.0))
        if isinstance(neg_temperature_exp, torch.Tensor): neg_temperature_exp = neg_temperature_exp.item()
        
        pos_avg_logits = outputs.get("pos_avg_logits", torch.tensor(0.0))
        if isinstance(pos_avg_logits, torch.Tensor): pos_avg_logits = pos_avg_logits.item()
        
        neg_avg_logits = outputs.get("neg_avg_logits", torch.tensor(0.0))
        if isinstance(neg_avg_logits, torch.Tensor): neg_avg_logits = neg_avg_logits.item()
        
        # Dual Stream Metrics (V5 and earlier)
        stream_pos_avg = outputs.get("stream_pos_avg", torch.tensor(0.0)).item()
        stream_neg_avg = outputs.get("stream_neg_avg", torch.tensor(0.0)).item()
        pos_win_rate = outputs.get("pos_win_rate", torch.tensor(0.0)).item()
        pos_router_rate = outputs.get("pos_router_rate", torch.tensor(0.0)).item()
        pos_bias = outputs.get("pos_bias", torch.tensor(0.0)).item()
        neg_bias = outputs.get("neg_bias", torch.tensor(0.0)).item()
        
        # V6 Feature Fusion Metrics
        alpha_mean = outputs.get("alpha_mean", torch.tensor(0.0))
        alpha_std = outputs.get("alpha_std", torch.tensor(0.0))
        beta_mean = outputs.get("beta_mean", torch.tensor(0.0))
        beta_std = outputs.get("beta_std", torch.tensor(0.0))
        if isinstance(alpha_mean, torch.Tensor):
            alpha_mean = alpha_mean.item()
        if isinstance(alpha_std, torch.Tensor):
            alpha_std = alpha_std.item()
        if isinstance(beta_mean, torch.Tensor):
            beta_mean = beta_mean.item()
        if isinstance(beta_std, torch.Tensor):
            beta_std = beta_std.item()
        
        if isinstance(pos_avg_logits, torch.Tensor):
            pos_avg_logits = pos_avg_logits.item()
        if isinstance(neg_avg_logits, torch.Tensor):
            neg_avg_logits = neg_avg_logits.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        if isinstance(loss_router, torch.Tensor):
            loss_router /= accum_iter
        loss_contrastive /= accum_iter
        # loss_scaler(loss, optimizer, parameters=model.parameters(),
        #             update_grad=(data_iter_step + 1) % accum_iter == 0)

        loss.backward()

        # Gradient clipping
        if (data_iter_step + 1) % accum_iter == 0:
            grad_clip_norm = getattr(args, 'grad_clip_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, norm_type=2.0)
            optimizer.step()
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_router=loss_router_value)
        metric_logger.update(loss_contrastive=loss_contrastive_value)
        metric_logger.update(logit_scale=logit_scale)
        metric_logger.update(logit_bias=logit_bias)
        metric_logger.update(pos_avg_logits=pos_avg_logits)
        metric_logger.update(attn_temperature_exp=attn_temperature_exp)
        metric_logger.update(loss_temperature_exp=loss_temperature_exp)
        metric_logger.update(neg_temperature_exp=neg_temperature_exp)
        metric_logger.update(neg_avg_logits=neg_avg_logits)
        
        # Log new metrics if they are non-zero (assuming they are active)
        if stream_pos_avg != 0 or stream_neg_avg != 0:
            metric_logger.update(stream_pos_avg=stream_pos_avg)
            metric_logger.update(stream_neg_avg=stream_neg_avg)
            metric_logger.update(pos_win_rate=pos_win_rate)
            metric_logger.update(pos_router_rate=pos_router_rate)
            metric_logger.update(pos_bias=pos_bias)
            metric_logger.update(neg_bias=neg_bias)
        
        # Log V6 Feature Fusion metrics
        if alpha_mean != 0 or beta_mean != 0:
            metric_logger.update(alpha_mean=alpha_mean)
            metric_logger.update(alpha_std=alpha_std)
            metric_logger.update(beta_mean=beta_mean)
            metric_logger.update(beta_std=beta_std)
            
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss_router', loss_router_value, epoch_1000x)
            log_writer.add_scalar('loss_contrastive', loss_contrastive_value, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar('logit_scale', logit_scale, epoch_1000x)
            log_writer.add_scalar('logit_bias', logit_bias, epoch_1000x)
            log_writer.add_scalar('pos_avg_logits', pos_avg_logits, epoch_1000x)
            log_writer.add_scalar('neg_avg_logits', neg_avg_logits, epoch_1000x)
            log_writer.add_scalar('attn_temperature_exp', attn_temperature_exp, epoch_1000x)
            log_writer.add_scalar('loss_temperature_exp', loss_temperature_exp, epoch_1000x)
            log_writer.add_scalar('neg_temperature_exp', neg_temperature_exp, epoch_1000x)
            
            if stream_pos_avg != 0 or stream_neg_avg != 0:
                log_writer.add_scalar('stream_pos_avg', stream_pos_avg, epoch_1000x)
                log_writer.add_scalar('stream_neg_avg', stream_neg_avg, epoch_1000x)
                log_writer.add_scalar('pos_win_rate', pos_win_rate, epoch_1000x)
                log_writer.add_scalar('pos_router_rate', pos_router_rate, epoch_1000x)
                log_writer.add_scalar('pos_bias', pos_bias, epoch_1000x)
                log_writer.add_scalar('neg_bias', neg_bias, epoch_1000x)
            
            # Log V6 Feature Fusion metrics
            if alpha_mean != 0 or beta_mean != 0:
                log_writer.add_scalar('alpha_mean', alpha_mean, epoch_1000x)
                log_writer.add_scalar('alpha_std', alpha_std, epoch_1000x)
                log_writer.add_scalar('beta_mean', beta_mean, epoch_1000x)
                log_writer.add_scalar('beta_std', beta_std, epoch_1000x)
        
        if 'debug' in args.note:
            break
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args=None, log_writer=None, epoch=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Validation:'

    # switch to evaluation mode
    model.eval()
    
    # Storage for classification metrics
    tp_counts = defaultdict(int)
    fp_counts = defaultdict(int)
    tn_counts = defaultdict(int)
    fn_counts = defaultdict(int)
    total_counts = defaultdict(int) # Valid samples per entity

    for batch in metric_logger.log_every(data_loader, 10, header):
        # Check if it is classification validation
        if batch.get("is_classification_val", False):
            # --- Classification Validation Logic ---
            images = batch["images"].to(device, non_blocking=True)
            gt_presence = batch["gt_presence"].to(device, non_blocking=True) # [B, 37]
            
            pos_input_ids = batch["pos_input_ids"].to(device, non_blocking=True) # [B*37, L]
            pos_attention_mask = batch["pos_attention_mask"].to(device, non_blocking=True)
            neg_input_ids = batch["neg_input_ids"].to(device, non_blocking=True) # [B*37, L]
            neg_attention_mask = batch["neg_attention_mask"].to(device, non_blocking=True)
            
            with torch.autocast("cuda", dtype=torch.bfloat16):
                # Using model.compute_logits to handle feature extraction, projection and similarity computation
                # This aligns with the demo notebook logic
                
                # compute_logits expects:
                # pixel_values: [B, 3, H, W]
                # encoded_key_phrases: list of dicts or dict with input_ids and attention_mask
                
                
                # 1. Vision Forward
                image_tokens = model.module.vision_encoder(images) if hasattr(model, 'module') else model.vision_encoder(images)
                image_tokens = model.module.vision_proj(image_tokens) if hasattr(model, 'module') else model.vision_proj(image_tokens)
                # image_tokens: [B, N_patch, D]
                
                # 2. Text Forward (Pos)
                pos_text_feats = model.module.text_encoder(input_ids=pos_input_ids, attention_mask=pos_attention_mask) if hasattr(model, 'module') else model.text_encoder(input_ids=pos_input_ids, attention_mask=pos_attention_mask)
                pos_text_feats = model.module.text_proj(pos_text_feats) if hasattr(model, 'module') else model.text_proj(pos_text_feats)
                # pos_text_feats: [B*37, 1, D] (TextEncoder returns [Batch, Num_Entities, D]) -> View to [B*37, D]
                pos_text_feats = pos_text_feats.view(-1, 512)
                
                # 3. Text Forward (Neg)
                neg_text_feats = model.module.text_encoder(input_ids=neg_input_ids, attention_mask=neg_attention_mask) if hasattr(model, 'module') else model.text_encoder(input_ids=neg_input_ids, attention_mask=neg_attention_mask)
                neg_text_feats = model.module.text_proj(neg_text_feats) if hasattr(model, 'module') else model.text_proj(neg_text_feats)
                neg_text_feats = neg_text_feats.view(-1, 512)
                
                # 4. Get Criterion and Params
                criterion = model.module.criterion if hasattr(model, 'module') else model.criterion
                
                # Reshape text feats: [B, 37, D]
                B, num_entities = gt_presence.shape
                pos_text_feats = pos_text_feats.view(B, num_entities, -1)
                neg_text_feats = neg_text_feats.view(B, num_entities, -1)
                
                pos_scores_list = []
                neg_scores_list = []
                
                for b_idx in range(B):
                    # Slice image: [1, 3, H, W]
                    img_b = images[b_idx].unsqueeze(0)
                    
                    # Slice texts: [1, 37] input_ids (need list of dicts)
                    # pos_input_ids is [B*37, L]. Slice [b*37 : (b+1)*37]
                    start = b_idx * num_entities
                    end = (b_idx + 1) * num_entities
                    
                    p_ids = pos_input_ids[start:end]
                    p_mask = pos_attention_mask[start:end]
                    
                    n_ids = neg_input_ids[start:end]
                    n_mask = neg_attention_mask[start:end]
                    
                    # Compute Pos Logits
                    # compute_logits expects encoded_key_phrases as list of dicts or dict
                    pos_enc = {"input_ids": p_ids, "attention_mask": p_mask}
                    
                    # Call model.compute_logits (handle DDP wrapper)
                    model_ref = model.module if hasattr(model, 'module') else model
                    out_pos = model_ref.compute_logits(pixel_values=img_b, encoded_key_phrases=pos_enc)
                    # logits: [1, 37] (assuming compute_logits returns [Batch_Img, Num_Text])
                    pos_logits_b = out_pos["logits"] 
                    
                    # Compute Neg Logits
                    neg_enc = {"input_ids": n_ids, "attention_mask": n_mask}
                    out_neg = model_ref.compute_logits(pixel_values=img_b, encoded_key_phrases=neg_enc)
                    neg_logits_b = out_neg["logits"]
                    
                    pos_scores_list.append(pos_logits_b)
                    neg_scores_list.append(neg_logits_b)
                
                # Concatenate
                pos_scores = torch.cat(pos_scores_list, dim=0) # [B, 37]
                neg_scores = torch.cat(neg_scores_list, dim=0) # [B, 37]
                
                # 4. Prediction: 1 if Pos > Neg, else 0
                # Using logits directly (sigmoid is monotonic, so comparison holds)
                preds = (pos_scores > neg_scores).long()
                
                # 5. Accumulate Metrics
                # Mask: ignore -100
                mask = (gt_presence != -100)
                
                for i in range(num_entities):
                    # Filter valid samples for this entity
                    valid_indices = mask[:, i]
                    if not valid_indices.any():
                        continue
                    
                    p = preds[valid_indices, i]
                    t = gt_presence[valid_indices, i]
                    
                    tp = ((p == 1) & (t == 1)).sum().item()
                    fp = ((p == 1) & (t == 0)).sum().item()
                    tn = ((p == 0) & (t == 0)).sum().item()
                    fn = ((p == 0) & (t == 1)).sum().item()
                    
                    tp_counts[i] += tp
                    fp_counts[i] += fp
                    tn_counts[i] += tn
                    fn_counts[i] += fn
                    total_counts[i] += valid_indices.sum().item()

        else:
            # --- Original Loss Logic ---
            # compute output
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(batch, device=device)
                loss = outputs["loss"]
                if "loss_contrastive" in outputs:
                    loss_contrastive = outputs["loss_contrastive"]
                else:
                    loss_contrastive = torch.tensor(0.0, device=device)
                # Support both loss_router (V5) and loss_ortho (V6)
                loss_router = outputs.get("loss_router", outputs.get("loss_ortho", torch.tensor(0.0, device=device)))
    
            loss_value = loss.item()
            loss_contrastive_value = loss_contrastive.item()
            loss_router_value = loss_router.item() if isinstance(loss_router, torch.Tensor) else loss_router
            pos_avg_logits = outputs.get("pos_avg_logits", 0.0)
            neg_avg_logits = outputs.get("neg_avg_logits", 0.0)
            attn_temperature_exp = outputs.get("attn_temperature_exp", 0.0)
            loss_temperature_exp = outputs.get("loss_temperature_exp", 0.0)
            neg_temperature_exp = outputs.get("neg_temperature_exp", 0.0)
            
            stream_pos_avg = outputs.get("stream_pos_avg", torch.tensor(0.0)).item()
            stream_neg_avg = outputs.get("stream_neg_avg", torch.tensor(0.0)).item()
            pos_win_rate = outputs.get("pos_win_rate", torch.tensor(0.0)).item()
            pos_router_rate = outputs.get("pos_router_rate", torch.tensor(0.0)).item()
            pos_bias = outputs.get("pos_bias", torch.tensor(0.0)).item()
            neg_bias = outputs.get("neg_bias", torch.tensor(0.0)).item()
            
            # V6 Feature Fusion Metrics
            alpha_mean = outputs.get("alpha_mean", torch.tensor(0.0))
            alpha_std = outputs.get("alpha_std", torch.tensor(0.0))
            beta_mean = outputs.get("beta_mean", torch.tensor(0.0))
            beta_std = outputs.get("beta_std", torch.tensor(0.0))
            if isinstance(alpha_mean, torch.Tensor):
                alpha_mean = alpha_mean.item()
            if isinstance(alpha_std, torch.Tensor):
                alpha_std = alpha_std.item()
            if isinstance(beta_mean, torch.Tensor):
                beta_mean = beta_mean.item()
            if isinstance(beta_std, torch.Tensor):
                beta_std = beta_std.item()
            
            if isinstance(pos_avg_logits, torch.Tensor):
                pos_avg_logits = pos_avg_logits.item()
            if isinstance(neg_avg_logits, torch.Tensor):
                neg_avg_logits = neg_avg_logits.item()
    
            metric_logger.update(loss=loss_value)
            metric_logger.update(loss_router=loss_router_value)
            metric_logger.update(loss_contrastive=loss_contrastive_value)
            metric_logger.update(pos_avg_logits=pos_avg_logits)
            metric_logger.update(neg_avg_logits=neg_avg_logits)
            metric_logger.update(attn_temperature_exp=attn_temperature_exp)
            metric_logger.update(loss_temperature_exp=loss_temperature_exp)
            metric_logger.update(neg_temperature_exp=neg_temperature_exp)
            
            if stream_pos_avg != 0 or stream_neg_avg != 0:
                metric_logger.update(stream_pos_avg=stream_pos_avg)
                metric_logger.update(stream_neg_avg=stream_neg_avg)
                metric_logger.update(pos_win_rate=pos_win_rate)
                metric_logger.update(pos_router_rate=pos_router_rate)
                metric_logger.update(pos_bias=pos_bias)
                metric_logger.update(neg_bias=neg_bias)
            
            # Log V6 Feature Fusion metrics
            if alpha_mean != 0 or beta_mean != 0:
                metric_logger.update(alpha_mean=alpha_mean)
                metric_logger.update(alpha_std=alpha_std)
                metric_logger.update(beta_mean=beta_mean)
                metric_logger.update(beta_std=beta_std)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    # --- Classification Metrics Aggregation ---
    if total_counts: # If we did classification validation
        # Synchronize counts across processes
        # Note: misc.all_reduce_sum assumes tensor input. We have dicts of ints.
        # We need to convert to tensor to reduce.
        
        # Determine num_entities from data if possible, or use max key
        all_keys = set(total_counts.keys())
        # We need all keys from all processes. This is tricky without knowing num_entities upfront.
        # However, num_entities is fixed (37).
        num_entities = 37 # Hardcoded or passed in args? Ideally passed. 
        # But for now let's assume keys are 0..36
        
        # Prepare tensors for reduction
        device = torch.device('cuda')
        counts_tensor = torch.zeros(4, num_entities, device=device, dtype=torch.long)
        
        for i in range(num_entities):
            counts_tensor[0, i] = tp_counts[i]
            counts_tensor[1, i] = fp_counts[i]
            counts_tensor[2, i] = tn_counts[i]
            counts_tensor[3, i] = fn_counts[i]
            
        torch.distributed.all_reduce(counts_tensor)
        
        # Compute F1 per entity
        f1_scores = []
        for i in range(num_entities):
            tp = counts_tensor[0, i].item()
            fp = counts_tensor[1, i].item()
            fn = counts_tensor[3, i].item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)
            
            # Update metric logger with per-entity F1 (optional, might be too many)
            # metric_logger.meters[f'f1_ent_{i}'].update(f1)
            
        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        print(f"Validation F1 Score: {avg_f1:.4f}")
        
        # Add to metric_logger so it gets returned and logged
        metric_logger.update(avg_f1=avg_f1)
        
        # Log to Tensorboard
        if log_writer is not None and epoch is not None:
            epoch_1000x = int(epoch * 1000)
            log_writer.add_scalar('valid_avg_f1', avg_f1, epoch_1000x)
            
            # Optionally log all entity F1s
            for i, f1 in enumerate(f1_scores):
                log_writer.add_scalar(f'valid_f1_ent_{i}', f1, epoch_1000x)

    if hasattr(metric_logger, 'loss'):
        print('* loss {losses.global_avg:.3f}'
              .format(losses=metric_logger.loss))

    if log_writer is not None and epoch is not None:
        epoch_1000x = int(epoch * 1000)
        
        if hasattr(metric_logger, 'loss'):
            log_writer.add_scalar('valid_loss', metric_logger.loss.global_avg, epoch_1000x)
        if hasattr(metric_logger, 'loss_router'):
            log_writer.add_scalar('valid_loss_router', metric_logger.loss_router.global_avg, epoch_1000x)
        if hasattr(metric_logger, 'loss_contrastive'):
            log_writer.add_scalar('valid_loss_contrastive', metric_logger.loss_contrastive.global_avg, epoch_1000x)
        if hasattr(metric_logger, 'pos_avg_logits'):
            log_writer.add_scalar('valid_pos_avg_logits', metric_logger.pos_avg_logits.global_avg, epoch_1000x)
        if hasattr(metric_logger, 'neg_avg_logits'):
            log_writer.add_scalar('valid_neg_avg_logits', metric_logger.neg_avg_logits.global_avg, epoch_1000x)
        if hasattr(metric_logger, 'attn_temperature_exp'):
            log_writer.add_scalar('valid_attn_temperature_exp', metric_logger.attn_temperature_exp.global_avg, epoch_1000x)
        if hasattr(metric_logger, 'loss_temperature_exp'):
            log_writer.add_scalar('valid_loss_temperature_exp', metric_logger.loss_temperature_exp.global_avg, epoch_1000x)
        if hasattr(metric_logger, 'neg_temperature_exp'):
            log_writer.add_scalar('valid_neg_temperature_exp', metric_logger.neg_temperature_exp.global_avg, epoch_1000x)
        
        if hasattr(metric_logger, 'stream_pos_avg'):
            log_writer.add_scalar('valid_stream_pos_avg', metric_logger.stream_pos_avg.global_avg, epoch_1000x)
            log_writer.add_scalar('valid_stream_neg_avg', metric_logger.stream_neg_avg.global_avg, epoch_1000x)
            log_writer.add_scalar('valid_pos_win_rate', metric_logger.pos_win_rate.global_avg, epoch_1000x)
            log_writer.add_scalar('valid_pos_router_rate', metric_logger.pos_router_rate.global_avg, epoch_1000x)
            log_writer.add_scalar('valid_pos_bias', metric_logger.pos_bias.global_avg, epoch_1000x)
            log_writer.add_scalar('valid_neg_bias', metric_logger.neg_bias.global_avg, epoch_1000x)
        
        # Log V6 Feature Fusion metrics
        if hasattr(metric_logger, 'alpha_mean'):
            log_writer.add_scalar('valid_alpha_mean', metric_logger.alpha_mean.global_avg, epoch_1000x)
            log_writer.add_scalar('valid_alpha_std', metric_logger.alpha_std.global_avg, epoch_1000x)
            log_writer.add_scalar('valid_beta_mean', metric_logger.beta_mean.global_avg, epoch_1000x)
            log_writer.add_scalar('valid_beta_std', metric_logger.beta_std.global_avg, epoch_1000x)
            
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_chexpert(model, device, data_loader, tokenized_queries, args=None, log_writer=None, epoch=None):
    """
    Run zero-shot CheXpert validation and compute macro AUROC.
    
    Args:
        model: Model to evaluate.
        device: Evaluation device.
        data_loader: CheXpertValDataset DataLoader, available on the main process.
        tokenized_queries: Pre-tokenized query text.
        args: Runtime arguments.
        log_writer: Tensorboard writer
        epoch: Current epoch.
    """
    # Run evaluation only on the main process.
    if data_loader is None or tokenized_queries is None:
        return {}
    
    model.eval()
    
    # Class list used for fallback predictions.
    CLASSES = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"
    ]
    
    all_preds = []
    all_targets = []
    
    # Unwrap DDP when needed.
    model_ref = model.module if hasattr(model, 'module') else model
    
    print(f"Start CheXpert inference on {len(data_loader.dataset)} images...")
    
    for batch in data_loader:
        pixel_values = batch["pixel_values"].to(device)
        targets = batch["target"].numpy()
        
        all_targets.extend(targets)
        
        try:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = model_ref.compute_logits(
                    pixel_values=pixel_values,
                    encoded_key_phrases=[tokenized_queries]
                )
                logits = outputs["logits"]
                all_preds.extend(logits.sigmoid().cpu().numpy())
        except Exception as e:
            print(f"Error during inference: {e}")
            # Add zero predictions to keep arrays aligned.
            for _ in range(pixel_values.shape[0]):
                all_preds.append(np.zeros((len(CLASSES),)))
    
    if len(all_preds) == 0:
        print("Warning: No predictions generated for CheXpert validation")
        return {"macro_auroc": 0.0}
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Keep targets and predictions aligned.
    min_len = min(len(all_targets), len(all_preds))
    all_targets = all_targets[:min_len]
    all_preds = all_preds[:min_len]
    
    # Compute macro AUROC.
    try:
        macro_auroc = roc_auc_score(all_targets, all_preds, average='macro')        
        # Record to tensorboard when available on the main process.
        if misc.is_main_process() and log_writer is not None and epoch is not None:
            epoch_1000x = int(epoch * 1000)
            log_writer.add_scalar('chexpert_macro_auroc', macro_auroc, epoch_1000x)
            log_writer.flush()
        
        return {"macro_auroc": macro_auroc}
    except Exception as e:
        print(f"Error calculating macro AUROC: {e}")
        return {"macro_auroc": 0.0}
