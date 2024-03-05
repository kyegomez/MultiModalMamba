import torch
import pytest
from mm_mamba.block import MultiModalMambaBlock


@pytest.fixture
def mmblock():
    return MultiModalMambaBlock(
        dim=64,
        depth=5,
        dropout=0.1,
        heads=4,
        d_state=16,
        image_size=64,
        patch_size=16,
        encoder_dim=64,
        encoder_depth=5,
        encoder_heads=4,
        fusion_method="mlp",
    )


def test_mmblock_initialization(mmblock):
    assert isinstance(mmblock, MultiModalMambaBlock)
    assert mmblock.dim == 64
    assert mmblock.depth == 5
    assert mmblock.dropout == 0.1
    assert mmblock.heads == 4
    assert mmblock.d_state == 16
    assert mmblock.image_size == 64
    assert mmblock.patch_size == 16
    assert mmblock.encoder_dim == 64
    assert mmblock.encoder_depth == 5
    assert mmblock.encoder_heads == 4
    assert mmblock.fusion_method == "mlp"


@pytest.mark.parametrize(
    "fusion_method", ["mlp", "concat", "add", "visual_expert"]
)
def test_mmblock_forward(mmblock, fusion_method):
    mmblock.fusion_method = fusion_method
    text = torch.randn(1, 16, 64)
    img = torch.randn(1, 3, 64, 64)
    out = mmblock(text, img)
    assert out.shape == text.shape


def test_mmblock_check_fusion_method(mmblock):
    mmblock.check_fusion_method()
    assert mmblock.fusion_method in [
        "mlp",
        "concat",
        "add",
        "visual_expert",
    ]
