import torch


def build_constant_lr_scheduler(config, optimizer):
    scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer,
        factor=config.constant_lr_schedule_factor,
        total_iters=config.train_steps,
    )
    return scheduler
