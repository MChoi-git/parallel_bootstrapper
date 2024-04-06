from argparse import Namespace

import pytest
import torch

from pboot.modules import (
    MLP,
    MHA,
    TransformerBlock,
    RotaryPositionEmbedding,
    apply_rotary_position_embeddings,
    VocabEmbedding,
    Transformer,
    TransformerLM,
)


@pytest.fixture
def config():
    config = Namespace()
    config.model_dim = 8
    config.vocab_size = 64
    config.batch_size = 2
    config.nheads = 2
    config.max_seq_len = 32
    config.mha_dropout = 0.
    config.mlp_dropout = 0.
    config.attention_dropout = 0.
    config.mlp_expansion_scale = 4
    config.nlayers = 2
    return config


@pytest.fixture
def convergence_config():
    config = Namespace()
    config.model_dim = 512
    config.vocab_size = 1024
    config.batch_size = 2
    config.nheads = 8
    config.nlayers = 12
    config.max_seq_len = 16
    config.mha_dropout = 0.
    config.mlp_dropout = 0.
    config.attention_dropout = 0.
    config.mlp_expansion_scale = 4
    config.lr = 1e-2
    config.num_steps = 100
    config.loss_threshold = 1e-3
    return config


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("torch_compile", [True, False])
def test_MLP(device, dtype, torch_compile, config):
    config.device = device
    config.dtype = dtype

    mlp = MLP(config)

    if torch_compile is True:
        mlp.compile()

    inputs = torch.randn((config.batch_size, config.model_dim), device=device, dtype=dtype)

    ret = mlp(inputs)

    assert ret.shape == inputs.shape


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("torch_compile", [True, False])
def test_MHA(device, dtype, torch_compile, config):
    config.device = device
    config.dtype = dtype

    mha = MHA(config)

    if torch_compile is True:
        mha.compile()

    inputs = torch.randn((config.batch_size, config.max_seq_len, config.model_dim), device=device, dtype=dtype)
    cos = torch.randn((config.batch_size, config.max_seq_len, config.nheads, config.model_dim // config.nheads), device=device, dtype=dtype) 
    sin = torch.randn((config.batch_size, config.max_seq_len, config.nheads, config.model_dim // config.nheads), device=device, dtype=dtype) 

    ret = mha(inputs, cos, sin)

    assert ret.shape == inputs.shape


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("torch_compile", [True, False])
def test_TransformerBlock(device, dtype, torch_compile, config):
    config.device = device
    config.dtype = dtype

    tb = TransformerBlock(config)

    if torch_compile is True:
        tb.compile()

    inputs = torch.randn((config.batch_size, config.max_seq_len, config.model_dim), device=device, dtype=dtype)
    cos = torch.randn((config.batch_size, config.max_seq_len, config.nheads, config.model_dim // config.nheads), device=device, dtype=dtype) 
    sin = torch.randn((config.batch_size, config.max_seq_len, config.nheads, config.model_dim // config.nheads), device=device, dtype=dtype) 

    ret = tb(inputs, cos, sin)

    assert ret.shape == inputs.shape


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("torch_compile", [True, False])
def test_RotaryPositionEmbeddings(device, dtype, torch_compile, config):
    config.device = device
    config.dtype = dtype

    rope = RotaryPositionEmbedding(config)

    def _forward_fn(q, k):
        cos, sin = rope(q)
        return apply_rotary_position_embeddings(q, k, cos, sin)
    if torch_compile is True:
        _forward_fn = torch.compile(_forward_fn)

    q = torch.randn((config.batch_size, config.max_seq_len, config.nheads, config.model_dim // config.nheads), device=device, dtype=dtype)
    k = torch.randn((config.batch_size, config.max_seq_len, config.nheads, config.model_dim // config.nheads), device=device, dtype=dtype)

    q_ret, k_ret = _forward_fn(q, k)

    assert q_ret.shape == q.shape
    assert k_ret.shape == k.shape


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("torch_compile", [True, False])
def test_VocabEmbeding(device, dtype, torch_compile, config):
    config.device = device
    config.dtype = dtype

    embed = VocabEmbedding(config)

    if torch_compile is True:
        embed.compile()

    inputs = torch.randint(0, config.vocab_size, size=(config.batch_size, config.max_seq_len), device=device)

    ret = embed(inputs)

    assert ret.shape == (config.batch_size, config.max_seq_len, config.model_dim)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("torch_compile", [False, True])
def test_Transformer(device, dtype, torch_compile, config):
    config.device = device
    config.dtype = dtype

    tf = Transformer(config)

    if torch_compile is True:
        tf.compile()

    inputs = torch.randint(0, config.vocab_size, size=(config.batch_size, config.max_seq_len), device=device)

    ret = tf(inputs)

    assert ret.shape == (config.batch_size, config.max_seq_len, config.vocab_size)
    

@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("torch_compile", [False, True])
def test_TransformerLM_convergence(device, dtype, torch_compile, convergence_config):
    torch.manual_seed(0)
    
    convergence_config.device = device
    convergence_config.dtype = dtype
    
    model = TransformerLM(convergence_config)

    if torch_compile is True:
        model.compile()


    inputs = torch.randint(0, convergence_config.vocab_size, size=(convergence_config.batch_size, convergence_config.max_seq_len), device=convergence_config.device)

    optim = torch.optim.Adam(model.parameters(), lr=convergence_config.lr)

    for i in range(convergence_config.num_steps):
        optim.zero_grad()

        loss = model(inputs)

        assert loss.isfinite()

        loss.backward()

        optim.step()

    assert loss.item() < convergence_config.loss_threshold
