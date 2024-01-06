from torch import Tensor, nn
from zeta import RMSNorm

from mm_mamba.block import MultiModalMambaBlock


class MMM(nn.Module):
    """
    MultiModalMamba model.

    Args:
        vocab_size (int): Size of the vocabulary.
        dim (int): Dimension of the dense vectors.
        depth (int): Number of layers in the model.
        dropout (float): Dropout probability.
        heads (int): Number of attention heads.
        d_state (int): Dimension of the state.
        image_size (int): Size of the input image.
        patch_size (int): Size of the image patch.
        encoder_dim (int): Dimension of the encoder.
        encoder_depth (int): Number of layers in the encoder.
        encoder_heads (int): Number of attention heads in the encoder.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Examples::
    import torch
    from mm_mamba.model import MMM

    x = torch.randint(0, 10000, (1, 224))
    img = torch.randn(1, 3, 224, 224)

    model = MMM(
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
    )

    out = model(x, img)
    print(out.shape)

    """

    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        dropout: float,
        heads: int,
        d_state: int,
        image_size: int,
        patch_size: int,
        encoder_dim: int,
        encoder_depth: int,
        encoder_heads: int,
        fusion_method: str = "mlp",
        *args,
        **kwargs,
    ):
        super(MMM, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth
        self.dropout = dropout
        self.heads = heads
        self.d_state = d_state
        self.image_size = image_size
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim
        self.encoder_depth = encoder_depth
        self.encoder_heads = encoder_heads

        # Transforms integer indices to dense vectors of fixed size
        self.embedding = nn.Embedding(vocab_size, dim)

        # MultiModalMambaBlock in a list
        self.layers = nn.ModuleList(
            [
                MultiModalMambaBlock(
                    dim,
                    depth,
                    dropout,
                    heads,
                    d_state,
                    image_size,
                    patch_size,
                    encoder_dim,
                    encoder_depth,
                    encoder_heads,
                    fusion_method,
                    *args,
                    **kwargs,
                )
            ]
        )

        # Normalization layer
        self.rmsnorm = RMSNorm(dim)
        self.norm = nn.LayerNorm(dim)

        # Linear layer
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embedding.weight

        # Projection for the img
        self.img_proj = nn.Linear(dim, dim)

    def forward(self, text: Tensor, img: Tensor) -> Tensor:
        """
        Forward pass of the MultiModalMamba model.

        Args:
            text (Tensor): Input text tensor.
            img (Tensor): Input image tensor.

        Returns:
            Tensor: Output logits.
        """
        x = self.embedding(text)

        for layer in self.layers:
            x = layer(x, img)  # + x
            # x = x + x

        x = self.norm(x)
        logits = self.lm_head(x)

        return logits
