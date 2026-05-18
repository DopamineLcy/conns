# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import shutil
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

import pretrain_datasets
import conns

from engine_pretrain import train_one_epoch, evaluate, evaluate_chexpert



def get_args_parser():
    parser = argparse.ArgumentParser('Pre-training')

    # Model parameters
    parser.add_argument('--model', default='CoNNSModel', type=str, metavar='MODEL',
                        help='Name of model to train (default: CoNNSModel)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--use_vision_cls_token', action='store_true',
                        help='Use vision cls token')
    parser.set_defaults(use_vision_cls_token=False)
    parser.add_argument('--proj_dim', type=int, default=512,
                        help='projection dimension (default: 512)')
    parser.add_argument('--num_hidden_layers', type=int, default=2,
                        help='number of hidden layers (default: 2)')
    parser.add_argument('--there_is_prob', type=float, default=0.5,
                        help='probability of using "there is" template (default: 0.5)')
    parser.add_argument('--rad_dino_output_layer', type=int, default=-1,
                        help='output layer of rad-dino (default: -1)')
    parser.add_argument('--use_extra_pos_embed', action='store_true',
                        help='Use extra pos embed')
    parser.set_defaults(use_extra_pos_embed=False)
    
    # Optimizer parameters
    parser.add_argument('--epochs', default=20, type=int,
                        help='Number of epochs (default: 20 from memory.txt)')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus, default: 256 from memory.txt)')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (absolute lr, default: 1e-4 from memory.txt)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR (deprecated, use warmup_iterations)')
    parser.add_argument('--warmup_iterations', type=int, default=5000, metavar='N',
                        help='iterations to warmup LR (default: 5000 from memory.txt)')
    parser.add_argument('--grad_clip_norm', type=float, default=1.0, metavar='N',
                        help='gradient clipping norm (default: 1.0 from memory.txt)')
    parser.add_argument('--init_logit_scale', type=float, default=np.log(10), metavar='N',
                        help='initial logit scale')
    parser.add_argument('--init_logit_bias', type=float, default=-10, metavar='N',
                        help='initial logit bias')

    # Dataset parameters
    parser.add_argument('--data_path', default='data/raw_dataset/MIMIC-CXR-JPG/files', type=str,
                        help='MIMIC-CXR-JPG files directory')
    parser.add_argument('--report_root', default='data/conns_training/reports_extract_concepts', type=str,
                        help='extracted report JSON directory')
    parser.add_argument('--metadata_csv', default='data/conns_training/mimic_conns_training.csv', type=str,
                        help='training metadata CSV')
    parser.add_argument('--metadata_csv_frontal', default='data/conns_training/mimic_conns_training_frontal.csv', type=str,
                        help='optional frontal-only training metadata CSV')
    parser.add_argument('--concepts_path', default='data/conns_training/concepts.json', type=str,
                        help='entity concept JSON')
    parser.add_argument('--yes_expressions_dir', default='data/conns_training/yes_expressions', type=str,
                        help='positive expression statistics directory')
    parser.add_argument('--no_expressions_dir', default='data/conns_training/no_expressions', type=str,
                        help='negative expression statistics directory')
    parser.add_argument('--chexpert_val_root', default='data/CheXpert', type=str,
                        help='optional CheXpert validation image root used during training')
    parser.add_argument('--chexpert_val_csv', default='data/CheXpert/val_labels.csv', type=str,
                        help='optional CheXpert validation label CSV used during training')
    parser.add_argument('--vision_model_path', default='external/rad-dino-maira-2', type=str,
                        help='Rad-DINO MAIRA-2 local directory')
    parser.add_argument('--text_model_path', default='external/BiomedVLP-CXR-BERT-specialized', type=str,
                        help='CXR-BERT local directory')
    parser.add_argument('--nli_model_path', default='cross-encoder/nli-deberta-v3-small', type=str,
                        help='NLI model path or Hugging Face repo id for hard negative mining')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--is_augmentation', action='store_true',
                        help='Use data augmentation')
    parser.set_defaults(is_augmentation=False)
    parser.add_argument('--use_counterfactual', action='store_true',
                        help='Use counterfactual text')
    parser.set_defaults(use_counterfactual=False)
    parser.add_argument('--special_class_sampling_prob', type=float, default=0.5, metavar='N',
                        help='sampling probability for special classes (normal lung transparency, etc.) to mitigate class imbalance (default: 0.5)')
    parser.add_argument('--attn_temperature', type=float, default=None, metavar='N',
                        help='attention temperature')
    parser.add_argument('--aug_degrees', type=int, default=20, metavar='N',
                        help='augmentation degrees')
    parser.add_argument('--aug_shear', type=int, default=10, metavar='N',
                        help='augmentation shear')
    parser.add_argument('--aug_translate', type=float, nargs=2, default=[0.1, 0.1], metavar='N',
                        help='augmentation translate')
    parser.add_argument('--aug_scale', type=float, nargs=2, default=[0.95, 1.05], metavar='N',
                        help='augmentation scale')
    parser.add_argument('--all_view', action='store_true',
                        help='use all view data')
    parser.set_defaults(all_view=False)
    parser.add_argument('--aug_prob', type=float, default=1.0, metavar='N',
                        help='augmentation probability')

    # Environment parameters
    parser.add_argument('--output_dir', default='./experiments',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./experiments',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save_freq', default=20, type=int)
    parser.add_argument('--from_begin', action='store_true',
                        help='train from epoch 0')
    parser.set_defaults(from_begin=False)
    parser.add_argument('--script', type=str, default='')
    parser.add_argument('--note', type=str, default='')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--eval_freq', default=1, type=int, help='frequency of evaluation')
    return parser


def main(args):
    print(args)
    # Initialize distributed training when RANK / WORLD_SIZE are set.
    misc.init_distributed_mode(args)
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    args.output_dir = os.path.join(args.output_dir, f'pretrain_MODEL{args.model}_EP{args.epochs}_WM{args.warmup_epochs}_LR{args.lr}_BS{eff_batch_size}_{args.note}')
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.script and os.path.exists(args.script):
        shutil.copy(args.script, args.output_dir)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    dataset_module = pretrain_datasets
    
    dataset_type = 'MIMIC-CoNNS'
    dataset = dataset_module.__dict__[dataset_type]
    dataset_train = dataset(args.data_path, is_train=True, args=args)
    
    # 1. Standard Validation Dataset (for Loss)
    dataset_valid = dataset(args.data_path, is_train=False, args=args)
    
    # CheXpert Validation Setup (Only on Main Process)
    chexpert_loader = None
    chexpert_queries = None
    if misc.is_main_process():
        print("Initializing CheXpert Validation Dataset...")
        try:
            if hasattr(dataset_module, 'CheXpertValDataset'):
                chexpert_ds = dataset_module.CheXpertValDataset(
                    data_root=args.chexpert_val_root,
                    csv_path=args.chexpert_val_csv,
                    image_processor=dataset_train.rad_dino_processor,
                )
                chexpert_loader = torch.utils.data.DataLoader(
                    chexpert_ds, 
                    batch_size=64, 
                    num_workers=4, 
                    pin_memory=True, 
                    drop_last=False
                )
                
                classes = chexpert_ds.classes
                queries_pos = [f"{c}" for c in classes]
                
                chexpert_queries = dataset_train.tokenizer(
                    queries_pos,
                    padding="max_length", 
                    truncation=True, 
                    max_length=128, 
                    return_tensors="pt"
                )
                print(f"CheXpert dataset initialized with {len(chexpert_ds)} images.")
            else:
                print("Warning: CheXpertValDataset not found in dataset module.")
        except Exception as e:
            print(f"Error initializing CheXpert dataset: {e}")


    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    
    sampler_valid = torch.utils.data.DistributedSampler(
        dataset_valid, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )
    print("Sampler_valid = %s" % str(sampler_valid))

    args.log_dir = os.path.join(args.output_dir, "logs")
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=dataset_train.collate_fn,
        prefetch_factor=2,
        persistent_workers=True
    )
    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, sampler=sampler_valid,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
        collate_fn=dataset_valid.collate_fn,
        prefetch_factor=2,
        persistent_workers=True
    )

    # define the model
    model = conns.__dict__[args.model](args=args)

    model.to(device)
    # model = torch.compile(model)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # Custom weight decay: exclude parameters based on memory.txt rules
    # exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    def exclude_from_weight_decay(n, p):
        """Exclude parameters from weight decay based on memory.txt rules"""
        return p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    
    def include_in_weight_decay(n, p):
        """Include parameters in weight decay"""
        return not exclude_from_weight_decay(n, p)
    
    # Collect all parameters statistics
    all_params = list(model_without_ddp.named_parameters())
    trainable_params = [(n, p) for n, p in all_params if p.requires_grad]
    frozen_params = [(n, p) for n, p in all_params if not p.requires_grad]
    
    def count_parameters(params):
        """Count total number of parameters"""
        return sum(p.numel() for _, p in params)
    
    total_params = count_parameters(all_params)
    trainable_count = count_parameters(trainable_params)
    frozen_count = count_parameters(frozen_params)
    
    # Helper function to print and write to log file
    log_lines = []
    def print_and_log(*args, **kwargs):
        """Print to console and collect lines for log file"""
        line = ' '.join(str(arg) for arg in args)
        print(*args, **kwargs)
        log_lines.append(line)
    
    # Print parameter statistics
    print_and_log("=" * 80)
    print_and_log("Parameter Statistics:")
    print_and_log("=" * 80)
    print_and_log(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print_and_log(f"Trainable parameters: {trainable_count:,} ({trainable_count/1e6:.2f}M, {100*trainable_count/total_params:.2f}%)")
    print_and_log(f"Frozen parameters: {frozen_count:,} ({frozen_count/1e6:.2f}M, {100*frozen_count/total_params:.2f}%)")
    print_and_log()
    
    # Create parameter groups
    params_with_wd = [(n, p) for n, p in trainable_params if include_in_weight_decay(n, p)]
    params_no_wd = [(n, p) for n, p in trainable_params if exclude_from_weight_decay(n, p)]
    
    param_groups = [
        {
            "params": [p for n, p in params_with_wd],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in params_no_wd],
            "weight_decay": 0.0,
        },
    ]
    
    # Print optimizer parameter groups details
    print_and_log("=" * 80)
    print_and_log("Optimizer Parameter Groups:")
    print_and_log("=" * 80)
    for i, group in enumerate(param_groups):
        group_params = [p for p in group["params"]]
        group_count = sum(p.numel() for p in group_params)
        print_and_log(f"Group {i+1}: weight_decay={group['weight_decay']}")
        print_and_log(f"  - Parameter count: {len(group_params):,}")
        print_and_log(f"  - Total parameters: {group_count:,} ({group_count/1e6:.2f}M)")
        print_and_log(f"  - Percentage of trainable: {100*group_count/trainable_count:.2f}%")
        
        # Print sample parameter names
        sample_names = [n for n, p in (params_with_wd if i == 0 else params_no_wd)]
        if len(sample_names) > 0:
            print_and_log(f"  - Sample parameter names:")
            for name in sample_names:
                print_and_log(f"      {name}")
        print_and_log()
    
    # Print frozen parameter samples
    if len(frozen_params) > 0:
        print_and_log("=" * 80)
        print_and_log("Frozen Parameters (sample):")
        print_and_log("=" * 80)
        for name, _ in frozen_params:
            print_and_log(f"  {name}")
        print_and_log()
    
    optimizer_type = None
    try:
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95), fused=True)
        optimizer_type = "fused"
        print_and_log("=" * 80)
        print_and_log("Optimizer: Using fused AdamW")
        print_and_log(f"  - Learning rate: {args.lr}")
        print_and_log(f"  - Betas: (0.9, 0.95)")
        print_and_log(f"  - Total optimizer parameters: {sum(sum(p.numel() for p in g['params']) for g in param_groups):,}")
        print_and_log("=" * 80)
    except Exception as e:
        print_and_log(f"Fused AdamW failed: {e}. Using standard AdamW")
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
        optimizer_type = "standard"
        print_and_log("=" * 80)
        print_and_log("Optimizer: Using standard AdamW")
        print_and_log(f"  - Learning rate: {args.lr}")
        print_and_log(f"  - Betas: (0.9, 0.95)")
        print_and_log(f"  - Total optimizer parameters: {sum(sum(p.numel() for p in g['params']) for g in param_groups):,}")
        print_and_log("=" * 80)
    
    # Write to log.txt file
    if args.output_dir and misc.is_main_process():
        log_file_path = os.path.join(args.output_dir, "log.txt")
        with open(log_file_path, mode="a", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

        
    # for F1 validation end ----------------------------
    
    # Calculate iterations per epoch for warmup scheduling
    args.iter_per_epoch = len(data_loader_train)
    print(f"Start training for {args.epochs} epochs")
    print(f"Iterations per epoch: {args.iter_per_epoch}")
    print(f"Warmup iterations: {args.warmup_iterations}")
    print(f"Total iterations: {args.epochs * args.iter_per_epoch}")
    
    min_val_loss = float('inf')
    max_val_auroc = -1.0  # Initialize max AUROC score

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        # if epoch > 10:
        #     break
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args,
            dataset=dataset_train
        )

        if args.output_dir:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, name="newest")

        if args.output_dir and (((epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs)):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, name="interval_save")

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if epoch % args.eval_freq == 0:
            # 1. Standard Loss Validation
            val_stats = evaluate(data_loader_valid, model, device, args, log_writer=log_writer, epoch=epoch)
            print(f"Loss of the network on the {len(dataset_valid)} validation images: {val_stats['loss']:.3f}")
            
            if val_stats["loss"] < min_val_loss:
                min_val_loss = val_stats["loss"]
                if args.output_dir:
                    misc.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch=epoch, name="best_val_loss")
            
            # Merge stats
            log_stats.update({**{f'validation_{k}': v for k, v in val_stats.items()}})

            # 2. CheXpert Zero-shot Classification Validation
            if chexpert_loader is not None and chexpert_queries is not None:
                chexpert_stats = evaluate_chexpert(
                    model, device, chexpert_loader, chexpert_queries, 
                    args, log_writer=log_writer, epoch=epoch
                )
                if "macro_auroc" in chexpert_stats:
                    macro_auroc = chexpert_stats["macro_auroc"]
                    print(f"CheXpert Validation Macro AUROC: {macro_auroc:.4f}")
                    if macro_auroc > max_val_auroc:
                        max_val_auroc = macro_auroc
                        if args.output_dir:
                            misc.save_model(
                                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                                loss_scaler=loss_scaler, epoch=epoch, name="best_val_auc")
                    
                    # Merge stats
                    log_stats.update({**{f'chexpert_{k}': v for k, v in chexpert_stats.items()}})
            else:
                 # If no loader (e.g. non-main process), we still want to keep code running smoothly
                 pass

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
