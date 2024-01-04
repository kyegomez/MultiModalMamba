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
