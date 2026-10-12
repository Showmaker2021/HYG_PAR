import time
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn as nn
from parco.models.hscl_v2 import CDHypergraphLayer
from parco.models.nn.transformer import TransformerBlock


def benchmark_speed():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    d = 128
    B = 64
    num_runs = 200
    warmup = 50

    print(f"{'M (Agents)':<12} | {'Flat MHA (ms)':<16} | {'CD-HG v2.1 (ms)':<16} | {'Ratio':<10}")
    print("-" * 60)

    for M in [10, 20, 50, 100]:
        K = 8 if M <= 50 else 16
        flat_layer = TransformerBlock(embed_dim=d, num_heads=8).to(device).eval()
        cd_layer = CDHypergraphLayer(d=d, K=K, num_heads=4, topk=2).to(device).eval()

        h = torch.randn(B, M, d, device=device)
        e_bar = torch.randn(B, M, d, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = flat_layer(h)
                _ = cd_layer(h, e_bar)
            if device.type == "cuda":
                torch.cuda.synchronize()

            # Benchmark Flat
            t0 = time.time()
            for _ in range(num_runs):
                _ = flat_layer(h)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_flat = (time.time() - t0) / num_runs * 1000.0

            # Benchmark CD-HG v2.1
            t0 = time.time()
            for _ in range(num_runs):
                _ = cd_layer(h, e_bar)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_cd = (time.time() - t0) / num_runs * 1000.0

        ratio = t_cd / t_flat
        print(f"{M:<12} | {t_flat:<16.3f} | {t_cd:<16.3f} | {ratio:<10.2f}x")


if __name__ == "__main__":
    benchmark_speed()
