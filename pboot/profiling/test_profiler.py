import pytest
import torch

from pboot.profiling import profile_function


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_profile_function(device):
    def _function(x):
        y = torch.randn_like(x)
        z = x * y
        return z

    profile_function(_function, torch.randn((64, 64), device=device))
