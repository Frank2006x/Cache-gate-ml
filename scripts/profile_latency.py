"""
Standalone Latency & Hardware Profiling Tool for Person 1 (Systems Lead).
Benchmarks wall-clock microsecond latency of SSIM/pHash gating vs. backward adaptation
passes across TTA models (TENT, CoTTA, MEMO) using high-precision timers and CUDA synchronization.
"""

import argparse
from pathlib import Path
import sys
import time
import torch
import numpy as np

# Add parent directory to path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from gating.similarity import pytorch_ssim, PerceptualHashSimilarity
from models.backbone import build_backbone
from models.tent import Tent
from models.cotta import Cotta
from models.memo import Memo


def benchmark_fn(fn, warmup: int = 15, iters: int = 100, is_cuda: bool = False):
    """Accurately measure execution time of a callable in milliseconds."""
    for _ in range(warmup):
        fn()
    if is_cuda:
        torch.cuda.synchronize()

    timings = []
    for _ in range(iters):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if is_cuda:
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(timings)), float(np.std(timings)), float(np.min(timings))


def main():
    parser = argparse.ArgumentParser(description="Profile Gating Overhead vs TTA Adaptation Passes")
    parser.add_argument("--device", type=str, default="auto", help="'cuda', 'cpu', or 'auto'")
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--warmup", type=int, default=15, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    is_cuda = device.type == "cuda"

    print("=" * 70)
    print("      SYSTEMS & HARDWARE PROFILER: CACHE-GATING VS BACKWARD PASS")
    print("=" * 70)
    print(f" Target Compute Device : {device}")
    if is_cuda:
        print(f" GPU Model Name        : {torch.cuda.get_device_name(0)}")
        print(f" CUDA Memory Allocated : {torch.cuda.memory_allocated(0)/(1024**2):.1f} MB")
    print(f" Backbone Architecture : {args.backbone}")
    print(f" Benchmark Iterations  : {args.iters} (Warmup: {args.warmup})")
    print("=" * 70)

    # 1. Prepare sample frames (1, 3, 32, 32)
    frame_a = torch.rand(1, 3, 32, 32, device=device)
    frame_b = torch.clamp(frame_a + torch.randn_like(frame_a) * 0.02, 0.0, 1.0)

    # 2. Benchmark Gating Kernels
    print("\n[1/3] Benchmarking Gating Kernels...")
    ssim_mean, ssim_std, ssim_min = benchmark_fn(
        lambda: pytorch_ssim(frame_a, frame_b), warmup=args.warmup, iters=args.iters, is_cuda=is_cuda
    )
    print(f"  -> PyTorch GPU/CPU SSIM   : {ssim_mean:6.3f} ms (±{ssim_std:.3f} ms, min: {ssim_min:.3f} ms)")

    phash = PerceptualHashSimilarity()
    phash_mean, phash_std, phash_min = benchmark_fn(
        lambda: phash(frame_a, frame_b), warmup=args.warmup, iters=args.iters, is_cuda=is_cuda
    )
    print(f"  -> Perceptual Hash (pHash): {phash_mean:6.3f} ms (±{phash_std:.3f} ms, min: {phash_min:.3f} ms)")

    # 3. Benchmark Forward Inference
    print("\n[2/3] Benchmarking Pure Forward Inference...")
    backbone = build_backbone(args.backbone, num_classes=10, pretrained=False, dataset="cifar10").to(device)
    tent_model = Tent(backbone, lr=1e-3).to(device)

    fwd_mean, fwd_std, _ = benchmark_fn(
        lambda: tent_model.forward(frame_a), warmup=args.warmup, iters=args.iters, is_cuda=is_cuda
    )
    print(f"  -> Forward Pass (Inference Only): {fwd_mean:6.3f} ms (±{fwd_std:.3f} ms)")

    # 4. Benchmark Backward Adaptation Passes
    print("\n[3/3] Benchmarking Adaptation Passes (Forward + Loss + Backward + Optimizer)...")
    tent_adapt, tent_std, _ = benchmark_fn(
        lambda: tent_model.adapt_and_forward(frame_a), warmup=args.warmup, iters=args.iters, is_cuda=is_cuda
    )
    tent_bwd_only = max(0.001, tent_adapt - fwd_mean)
    print(f"  -> TENT Adaptation Pass        : {tent_adapt:6.3f} ms (Backward-only: {tent_bwd_only:6.3f} ms)")

    backbone_cotta = build_backbone(args.backbone, num_classes=10, pretrained=False, dataset="cifar10").to(device)
    cotta_model = Cotta(backbone_cotta, lr=1e-3).to(device)
    cotta_adapt, cotta_std, _ = benchmark_fn(
        lambda: cotta_model.adapt_and_forward(frame_a), warmup=args.warmup, iters=args.iters, is_cuda=is_cuda
    )
    print(f"  -> CoTTA Adaptation Pass       : {cotta_adapt:6.3f} ms (Dual Teacher-Student + EMA)")

    backbone_memo = build_backbone(args.backbone, num_classes=10, pretrained=False, dataset="cifar10").to(device)
    memo_model = Memo(backbone_memo, lr=1e-3, num_aug=3).to(device)
    memo_adapt, memo_std, _ = benchmark_fn(
        lambda: memo_model.adapt_and_forward(frame_a), warmup=args.warmup, iters=args.iters, is_cuda=is_cuda
    )
    print(f"  -> MEMO Adaptation Pass        : {memo_adapt:6.3f} ms (Multi-view Augmentations)")

    # 5. Breakeven & Theoretical Speedup Summary
    print("\n" + "=" * 70)
    print("                 FINAL HARDWARE PROFILING RESULTS")
    print("=" * 70)
    overhead_pct = (ssim_mean / tent_bwd_only) * 100.0
    r_breakeven = (ssim_mean / tent_bwd_only) * 100.0

    print(f"  PyTorch SSIM Gating Overhead : {ssim_mean:6.3f} ms")
    print(f"  TENT Isolated Backward Cost  : {tent_bwd_only:6.3f} ms")
    print(f"  Gating Overhead / Backprop   : {overhead_pct:6.2f} %")
    print(f"  Breakeven Hit Ratio          : {r_breakeven:6.2f} %")
    print("-" * 70)
    print("  WHAT THIS MEANS FOR YOUR PAPER:")
    if r_breakeven < 10.0:
        print(f"  ✓ Outstanding result! If only >{r_breakeven:.1f}% of video frames are redundant,")
        print("    Cache-Gating provides a NET WALL-CLOCK ACCELERATION to the entire system.")
    else:
        print(f"  Breakeven at {r_breakeven:.1f}% redundancy.")
    print("=" * 70)


if __name__ == "__main__":
    main()
