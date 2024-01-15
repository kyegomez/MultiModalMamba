import torch  # Import the torch library

# Import the MultiModalMamba model from the mm_mamba module
from mm_mamba import MultiModalMamba

# Generate a random tensor 'x' of size (1, 224) with random elements between 0 and 10000
x = torch.randint(0, 10000, (1, 196))

# Generate a random image tensor 'img' of size (1, 3, 224, 224)
img = torch.randn(1, 3, 224, 224)

# Audio tensor 'aud' of size 2d
aud = torch.randn(1, 224)

# Video tensor 'vid' of size 5d - (batch_size, channels, frames, height, width)
vid = torch.randn(1, 3, 16, 224, 224)

# Create a MultiModalMamba model object with the following parameters:
model = MultiModalMamba(
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
    post_fuse_norm=True,
)

# Pass the tensor 'x' and 'img' through the model and store the output in 'out'
out = model(x, img, aud, vid)

# Print the shape of the output tensor 'out'
print(out.shape)
