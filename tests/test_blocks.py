import pytest
import torch
from mm_mamba.block import MultiModalMambaBlock

@pytest.fixture
def model():
    return MultiModalMambaBlock(
        dim = 64,
        depth = 5,
        dropout = 0.1,
        heads = 4,
        d_state = 16,
        image_size = 64,
        patch_size = 16,
        encoder_dim = 64,
        encoder_depth = 5,
        encoder_heads = 4
    )

@pytest.fixture
def inputs():
    x = torch.randn(1, 16, 64)
    y = torch.randn(1, 3, 64, 64)
    return x, y

def test_forward(model, inputs):
    x, y = inputs
    output = model(x, y)
    assert output.shape == x.shape

def test_model_parameters(model):
    for param in model.parameters():
        assert param.requires_grad
        assert param.dim() > 0