#!/usr/bin/env python3
"""
简单的MSE测试
"""

import torch
import numpy as np

# 测试1：正常归一化数据的MSE
print("Test 1: Normalized data MSE")
a = torch.randn(2, 2048, 3) * 0.5  # 范围大约 [-1.5, 1.5]
b = torch.randn(2, 2048, 3) * 0.5
mse = torch.nn.functional.mse_loss(a, b)
print(f"  Data range: a=[{a.min():.3f}, {a.max():.3f}], b=[{b.min():.3f}, {b.max():.3f}]")
print(f"  MSE: {mse:.6f}")
print(f"  MSE (millions): {mse:.0f}")

# 测试2：如果数据没有归一化
print("\nTest 2: Unnormalized data MSE")
scale = 13.92  # 从你的数据中看到的scale
a_unnorm = a / 0.0718  # 逆归一化
b_unnorm = b / 0.0718
mse_unnorm = torch.nn.functional.mse_loss(a_unnorm, b_unnorm)
print(f"  Data range: a=[{a_unnorm.min():.3f}, {a_unnorm.max():.3f}]")
print(f"  MSE: {mse_unnorm:.6f}")
print(f"  MSE (millions): {mse_unnorm/1e6:.1f}M")

# 测试3：检查你看到的MSE值
print("\nTest 3: Your MSE value analysis")
your_mse = 88350396
print(f"  Your MSE: {your_mse}")
print(f"  Square root: {np.sqrt(your_mse):.1f}")
print(f"  This suggests data range of: ±{np.sqrt(your_mse/3):.1f}")

# 测试4：如果生成的是纯噪声且范围很大
print("\nTest 4: Large noise MSE")
large_noise = torch.randn(2, 2048, 3) * 100  # 大噪声
normal_data = torch.randn(2, 2048, 3) * 0.5
mse_large = torch.nn.functional.mse_loss(large_noise, normal_data)
print(f"  Noise range: [{large_noise.min():.1f}, {large_noise.max():.1f}]")
print(f"  MSE: {mse_large:.0f}")
print(f"  MSE (millions): {mse_large/1e6:.1f}M")

# 结论
print("\n" + "="*60)
print("CONCLUSION:")
print("Your MSE of 88M suggests the generated points have range ~±5000")
print("This means the diffusion model is generating extremely large values!")
print("Possible causes:")
print("1. Model weights exploded during training")
print("2. Diffusion sampling is not clamping values")
print("3. Some numerical instability in the generation process")