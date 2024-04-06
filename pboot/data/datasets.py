from argparse import Namespace

import torch
from torch import Tensor

from pboot.config import get_config


class SyntheticMapDataset(torch.utils.data.Dataset):
    def __init__(self, config: Namespace):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.seq_len = config.seq_len
        self.seed = config.seed
        self.num_samples = config.num_samples

        self.rng = torch.Generator(device="cpu")
        self.rng = self.rng.manual_seed(self.seed)

        if config.overfit is True:
            self.overfit_batch = torch.randint(
                low=0,
                high=self.vocab_size,
                size=(self.seq_len,),
                generator=self.rng,
            )
        
    def __getitem__(self, i: int):
        if hasattr(self, "overfit_batch"):
            return self.overfit_batch

        sequence = torch.randint(
            low=0,
            high=self.vocab_size,
            size=(self.seq_len,),
            generator=self.rng,
        )

        return sequence

    def __len__(self):
        return self.num_samples


def tensor_collate_fn(batch_data: list[Tensor]) -> Tensor:
    batch_tensor = torch.stack(batch_data, dim=0)
    return batch_tensor


def build_data_iterable(
    dataset: torch.utils.data.Dataset,
    sampler: torch.utils.data.Sampler,
    collate_fn: callable,
    batch_size: int,
    num_workers: int,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
