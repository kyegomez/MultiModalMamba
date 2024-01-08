import timeit
import torch
from torch import nn
from zeta.nn import MambaBlock
from zeta.structs import Transformer, Decoder


gpt4 = Transformer(
    num_tokens = 50000,
    max_seq_len = 2048,
    attn_layers = Decoder(
        dim = 12288,
        depth = 96,
        heads = 96,
        attn_dim_head = 128
    )
).cuda()

def benchmark_mamba_block():
    mamba_block = MambaBlock(dim=2048, depth=6, d_state=64)
    input = torch.randn(1, 64, 2048)
    start_time = timeit.default_timer()
    mamba_block(input)
    end_time = timeit.default_timer()
    return end_time - start_time

def benchmark_gpt4_transformer():
    input = torch.randint(0, 50000, (1, 2048))
    start_time = timeit.default_timer()
    gpt4(input)
    end_time = timeit.default_timer()
    return end_time - start_time

mamba_time = benchmark_mamba_block()
gpt4_time = benchmark_gpt4_transformer()

print(f"MambaBlock execution time: {mamba_time} seconds")
print(f"gpt4 Transformer execution time: {gpt4_time} seconds")

# Calculate and print the difference in execution times
time_difference = abs(mamba_time - gpt4_time)
print(f"Difference in execution time: {time_difference} seconds")