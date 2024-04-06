import pytest
import torch


from pboot.data import CheckpointableDistributedSampler


def test_CheckpointableDistributedSampler_deterministic():
    dataset_length = 100
    rank = 0
    world_size = 1
    sampler = CheckpointableDistributedSampler(
        dataset_length, rank, world_size
    )

    indices = list(iter(sampler))

    new_sampler = CheckpointableDistributedSampler(
        dataset_length, rank, world_size
    )
    assert list(iter(new_sampler)) == indices


def test_CheckpointableDistributedSampler_resume():
    dataset_length = 100
    rank = 0
    world_size = 1
    num_samples_to_consume = 41

    sampler = CheckpointableDistributedSampler(
        dataset_length, rank, world_size
    )

    # All indices for epoch
    full_indices = list(iter(sampler))

    # Indices up until checkpoint
    iter_sampler = iter(sampler)
    pre_indices = [next(iter_sampler) for _ in range(num_samples_to_consume)]

    # Indices after checkpoint
    resume_sampler = CheckpointableDistributedSampler(
        dataset_length, rank, world_size, consumed_samples=num_samples_to_consume
    )
    iter_sampler = iter(resume_sampler)
    post_indices = [next(iter_sampler) for _ in range(dataset_length - num_samples_to_consume)]

    assert full_indices == pre_indices + post_indices


@pytest.mark.parametrize("dset_len", [100, 99])
@pytest.mark.parametrize("consumed_samples", [0, 50, 99, 100, 101])
def test_CheckpointableDistributedSampler_multiprocess(dset_len, consumed_samples):
    world_size = 2

    sampler_0 = CheckpointableDistributedSampler(
        dset_len, 0, world_size, consumed_samples=consumed_samples
    )
    indices_0 = list(iter(sampler_0))

    sampler_1 = CheckpointableDistributedSampler(
        dset_len, 1, world_size, consumed_samples=consumed_samples
    )
    indices_1 = list(iter(sampler_1))

    assert len(indices_0) == len(indices_1)
    assert set(indices_0).isdisjoint(set(indices_1))
