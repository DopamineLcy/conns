# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math


def adjust_learning_rate(optimizer, epoch_or_iter, args):
    """Decay the learning rate with half-cycle cosine after warmup
    Args:
        epoch_or_iter: Can be epoch (float) or iteration (int) depending on warmup_mode
    """
    if hasattr(args, 'warmup_iterations') and args.warmup_iterations > 0 and hasattr(args, 'iter_per_epoch'):
        # Warmup based on iterations
        # epoch_or_iter is the current iteration number
        current_iter = int(epoch_or_iter)
        if current_iter < args.warmup_iterations:
            lr = args.lr * current_iter / args.warmup_iterations
        else:
            # Cosine decay after warmup
            total_iters = args.epochs * args.iter_per_epoch
            progress = (current_iter - args.warmup_iterations) / max(1, total_iters - args.warmup_iterations)
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1. + math.cos(math.pi * progress))
    else:
        # Warmup based on epochs (backward compatibility)
        if epoch_or_iter < args.warmup_epochs:
            lr = args.lr * epoch_or_iter / args.warmup_epochs 
        else:
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch_or_iter - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
