from argparse import Namespace

import pytest
import torch

from pboot.data import SyntheticMapDataset


@pytest.fixture
def dataset_config():
    config = Namespace()
    config.vocab_size = 32
    config.seq_len = 16
    config.seed = 0
    config.num_samples = 50
    config.batch_size = 8
    config.num_workers = 4
    config.overfit = False
    return config


def test_SyntheticMapDataset_shape(dataset_config):
    dataset = SyntheticMapDataset(dataset_config)

    assert len(dataset) == dataset_config.num_samples
    assert dataset[3].shape == (dataset_config.seq_len,)


def test_SyntheticMapDataset_determinstic(dataset_config):
    torch.manual_seed(0)

    dset = SyntheticMapDataset(dataset_config)
    # NOTE: Calling getitem on both dsets at once does not produce idential
    #       samples.
    samples = [dset[i] for i in range(len(dset))]

    new_dset = SyntheticMapDataset(dataset_config)
    for i in range(len(new_dset)):
        assert torch.equal(samples[i], new_dset[i])


def test_synthetic_dataset_pipeline(dataset_config):
    from pboot.data import (
        CheckpointableDistributedSampler,
        tensor_collate_fn,
        build_data_iterable,
    )

    dset = SyntheticMapDataset(dataset_config)

    def _sample_rank(r, world_size):
        sampler = CheckpointableDistributedSampler(
            len(dset), rank=r, world_size=world_size,
        )
        dl = build_data_iterable(
            dset, sampler, tensor_collate_fn, dataset_config.batch_size, dataset_config.num_workers
        )
        dl = iter(dl)
        samples = []
        for _ in range(len(dl)):
            samples.extend(torch.chunk(next(dl), dataset_config.batch_size, dim=0))
        return set(samples)

    samples_rank0 = _sample_rank(0, 2)
    samples_rank1 = _sample_rank(1, 2)

    assert samples_rank0.isdisjoint(samples_rank1)
