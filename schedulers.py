import numpy as np


def get_policy(name):
    if name is None:
        return constant_lr

    out_dict = {
        "constant_lr": constant_lr,
        "linear_lr": linear_lr,
        "sqrt_lr": sqrt_lr,
    }

    return out_dict[name]


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def constant_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        lr = args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def sqrt_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if iteration <= args.stop_iteration:
            lr = args.lr
        else:
            lr = args.lr * np.sqrt(args.stop_iteration) / np.sqrt(iteration)
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def linear_lr(optimizer, args, max_iterations, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if iteration <= args.stop_iteration:
            lr = args.lr
        else:
            lr = -0.9 * args.lr / (max_iterations - args.stop_iteration) * (iteration - args.stop_iteration) + args.lr
        assign_learning_rate(optimizer, lr)
        return lr
    
    return _lr_adjuster


def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length
