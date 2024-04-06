import torch
from torch.optim import Adam


TorchAdam = Adam


def build_torch_adam(config, model):
    adam = TorchAdam(
        model.parameters(),
        config.lr,
        (config.beta1, config.beta2),
        config.optim_eps,
        config.weight_decay,
        fused=config.use_fused_adam,
    )
    return adam
