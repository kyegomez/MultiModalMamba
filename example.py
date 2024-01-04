# Import the necessary libraries
import torch
from mm_mamba import MultiModalMambaBlock

# Create some random input tensors
x = torch.randn(
    1, 16, 64
)  # Tensor with shape (batch_size, sequence_length, feature_dim)
y = torch.randn(
    1, 3, 64, 64
)  # Tensor with shape (batch_size, num_channels, image_height, image_width)

# Create an instance of the MultiModalMambaBlock model
model = MultiModalMambaBlock(
    dim=64,  # Dimension of the token embeddings
    depth=5,  # Number of transformer layers
    dropout=0.1,  # Dropout probability
    heads=4,  # Number of attention heads
    d_state=16,  # Dimension of the state embeddings
    image_size=64,  # Size of the input image
    patch_size=16,  # Size of each image patch
    encoder_dim=64,  # Dimension of the encoder token embeddings
    encoder_depth=5,  # Number of encoder transformer layers
    encoder_heads=4,  # Number of encoder attention heads
)

# Pass the input tensors through the model
out = model(x, y)

# Print the shape of the output tensor
print(out.shape)
