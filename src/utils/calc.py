import numpy as np
import torch as th

def compute_q_values(rewards, discount_factor=0.99):
    batch_size, timestep, _ = rewards.shape
    q_values = th.zeros_like(rewards)

    for i in range(timestep - 1, -1, -1):
        if i == timestep - 1:
            q_values[:, i, :] = rewards[:, i, :]
        else:
            q_values[:, i, :] = rewards[:, i, :] + discount_factor * q_values[:, i + 1, :]

    return q_values

def count_total_parameters(model: th.nn.Module, prefix=''):
    n_total = sum(param.numel() for param in model.parameters())
    n_trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print("Model Parameter Stats:")
    print(f"Trainable: {n_trainable:,}, Total: {n_total:,}")
    print("=" * 80)
    for name, param in model.named_parameters(prefix=prefix):
        print(f"{param.numel():,} \t parameters -> {name}")
    print("=" * 80)
    
