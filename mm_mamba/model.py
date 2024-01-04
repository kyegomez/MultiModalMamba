from torch import nn, Tensor
from torch import nn, Tensor
from zeta import RMSNorm
from mm_mamba import MultiModalMambaBlock


class MMM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        depth: int,
        dropout,
        heads,
        d_state,
        *args,
        **kwargs,
    ):
        super(MMM, self).__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.depth = depth

        self.embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            [
                MultiModalMambaBlock(
                    dim,
                    depth,
                    dropout,
                    heads,
                    d_state,
                )
            ]
        )
        self.rmsnorm = RMSNorm(dim)

        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def forward(self, text: Tensor, img: Tensor) -> Tensor:
        self.embedding(text)
