import torch
import pytest
from mm_mamba.model import MMM


@pytest.fixture
def mmm():
    return MMM(
        vocab_size=10000,
        dim=512,
        depth=6,
        dropout=0.1,
        heads=8,
        d_state=512,
        image_size=224,
        patch_size=16,
        encoder_dim=512,
        encoder_depth=6,
        encoder_heads=8,
        fusion_method="mlp",
        return_embeddings=False,
    )


def test_mmm_initialization(mmm):
    assert isinstance(mmm, MMM)
    assert mmm.vocab_size == 10000
    assert mmm.dim == 512
    assert mmm.depth == 6
    assert mmm.dropout == 0.1
    assert mmm.heads == 8
    assert mmm.d_state == 512
    assert mmm.image_size == 224
    assert mmm.patch_size == 16
    assert mmm.encoder_dim == 512
    assert mmm.encoder_depth == 6
    assert mmm.encoder_heads == 8
    assert mmm.fusion_method == "mlp"
    assert mmm.return_embeddings is False


def test_mmm_forward(mmm):
    text = torch.randint(0, 10000, (1, 224))
    img = torch.randn(1, 3, 224, 224)
    out = mmm(text, img)
    assert out.shape == (1, 224, 10000)


def test_mmm_return_embeddings(mmm):
    mmm.return_embeddings = True
    text = torch.randint(0, 10000, (1, 224))
    img = torch.randn(1, 3, 224, 224)
    out = mmm(text, img)
    assert out.shape == (1, 224, 512)


@pytest.mark.parametrize(
    "fusion_method", ["mlp", "concat", "add", "visual_expert"]
)
def test_mmm_forward_with_different_fusion_methods(
    mmm, fusion_method
):
    mmm.fusion_method = fusion_method
    text = torch.randint(0, 10000, (1, 224))
    img = torch.randn(1, 3, 224, 224)
    out = mmm(text, img)
    assert out.shape == (1, 224, 10000)


@pytest.mark.parametrize(
    "fusion_method", ["mlp", "concat", "add", "visual_expert"]
)
def test_mmm_forward_with_different_fusion_methods_2(
    mmm, fusion_method
):
    mmm.fusion_method = fusion_method
    text = torch.randint(0, 10000, (1, 224))
    img = torch.randn(1, 3, 224, 224)
    out = mmm(text, img)
    assert (
        out.shape == (1, 224, 10000)
        if not mmm.return_embeddings
        else (1, 224, 512)
    )
