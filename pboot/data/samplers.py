import torch


class CheckpointableDistributedSampler(torch.utils.data.Sampler):
    """Distributed random sampler which can be checkpointed.

    Can be used to resume sampling given the previously consumed samples by all
    workers. No batching is performed.
    """
    def __init__(
        self,
        dataset_length: int,
        rank: int,
        world_size: int,
        seed: int = 0,
        consumed_samples: int = 0,
    ):
        self.dataset_length = dataset_length
        self.rank = rank
        self.world_size = world_size
        self.seed = seed

        # Drop last to calculate samples per epoch
        self.total_consumed_samples = consumed_samples
        self.total_samples_per_epoch = (dataset_length // world_size) * world_size
        
        self._update_stats()

    def _update_stats(self):
        self.epoch = self.total_consumed_samples // self.total_samples_per_epoch
        self.num_samples_into_epoch = self.total_consumed_samples % self.total_samples_per_epoch

        # Drop last: Bump into next epoch if num remaining samples < world size
        if self.total_samples_per_epoch - self.num_samples_into_epoch < self.world_size:
            self.epoch += 1
            self.total_consumed_samples = self.epoch * self.total_samples_per_epoch
            self.num_samples_into_epoch = 0

        self.num_samples = (
            (self.total_samples_per_epoch - self.num_samples_into_epoch) //
            self.world_size
        )

    def __iter__(self):
        rng = torch.Generator(device="cpu")
        rng.manual_seed(self.seed + self.epoch)

        # All indices in epoch
        indices = torch.randperm(self.total_samples_per_epoch, generator=rng).tolist()

        # Resume from checkpoint index
        indices = indices[self.num_samples_into_epoch: self.total_samples_per_epoch]

        # Chunk remaining data between replicas
        indices = indices[self.rank * self.num_samples: self.rank * self.num_samples + self.num_samples]

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
