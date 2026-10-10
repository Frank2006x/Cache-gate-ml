"""
Main execution loop for Cache-Gated Continuous Test-Time Adaptation.
Orchestrates streaming data generation, gating decisions, TTA model updates, and metrics profiling.
"""

import argparse
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn

# Ensure local packages are importable regardless of current working directory
_repo_dir = Path(__file__).resolve().parent
if str(_repo_dir) not in sys.path:
    sys.path.insert(0, str(_repo_dir))
_parent_dir = _repo_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from data.dataset_loader import build_dataset, CORRUPTIONS
from data.stream_generator import CorrelatedStreamDataset
from models.backbone import build_backbone
from models.source import Source
from models.tent import Tent
from models.eata import Eata
from models.sar import Sar
from models.cotta import Cotta
from models.dua import Dua
from models.memo import Memo
from gating.cache_gate import CacheGate
from utils.metrics import TTAMetricsTracker
from utils.logger import ExperimentLogger


def set_seed(seed: int = 42) -> None:
    """Set random seeds for determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Algorithmic Cache-Gating for Compute-Efficient Continuous TTA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model and TTA settings
    parser.add_argument(
        "--model",
        type=str,
        default="tent",
        choices=["source", "tent", "eata", "sar", "cotta", "dua", "memo"],
        help="TTA adaptation algorithm or source baseline (7 models supported)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50"],
        help="Vision backbone architecture",
    )
    parser.add_argument("--pretrained", action="store_true", help="Load torchvision pretrained weights")
    parser.add_argument("--lr", type=float, default=1e-3, help="TTA adaptation learning rate")

    # Dataset and stream settings
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagenet", "dummy"])
    parser.add_argument("--data_dir", type=str, default="./data", help="Root directory for datasets")
    parser.add_argument(
        "--corruption",
        type=str,
        default="gaussian_noise",
        choices=["none"] + CORRUPTIONS,
        help="Corruption type (CIFAR-10-C / ImageNet-C)",
    )
    parser.add_argument("--severity", type=int, default=5, choices=[1, 2, 3, 4, 5], help="Corruption severity level")
    parser.add_argument("--stream_length", type=int, default=1000, help="Number of frames in the continuous stream")
    parser.add_argument("--repeat_prob", type=float, default=0.6, help="Temporal persistence / repetition probability")
    parser.add_argument("--noise_std", type=float, default=0.01, help="Sensor noise std injected into repeated frames")
    parser.add_argument("--jitter_shift", type=int, default=0, help="Max pixel translation for camera micro-shake (default: 0 for 32x32 CIFAR)")

    # Gating settings
    parser.add_argument(
        "--gating",
        type=str,
        default="ssim",
        choices=["none", "ssim", "phash", "l1", "l2", "cosine"],
        help="Cache gating metric ('none' runs full continuous adaptation without gating)",
    )
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold tau for cache hit")
    parser.add_argument("--cache_capacity", type=int, default=1, help="Number of keyframe slots in cache")
    parser.add_argument(
        "--cache_policy",
        type=str,
        default="keyframe",
        choices=["keyframe", "fifo", "ema"],
        help="Cache replacement policy",
    )

    # Execution and logging settings
    parser.add_argument("--device", type=str, default="auto", help="'cuda', 'cpu', or 'auto'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment identifier name")
    parser.add_argument("--log_dir", type=str, default="./runs", help="Directory for experiment outputs")
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--print_freq", type=int, default=50, help="Progress printing interval")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap evaluation to first N samples (for test runs)")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Determine compute device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    corr = None if args.corruption == "none" else args.corruption
    exp_name = args.exp_name or f"{args.model}_{args.backbone}_{corr or 'clean'}_{args.gating}"
    if args.gating != "none":
        exp_name += f"_tau{args.threshold}"

    print("=" * 65)
    print("   CACHE-GATED CONTINUOUS TEST-TIME ADAPTATION")
    print("=" * 65)
    print(f" Device          : {device}")
    print(f" Model / TTA     : {args.model.upper()} (Backbone: {args.backbone})")
    print(f" Dataset         : {args.dataset.upper()} | Corruption: {corr or 'clean'} (Sev: {args.severity})")
    print(f" Temporal Stream : {args.stream_length} frames | Repeat Prob: {args.repeat_prob}")
    print(f" Cache Gating    : {args.gating.upper()} (Threshold: {args.threshold}, Policy: {args.cache_policy})")
    print("=" * 65)

    # 1. Build Base Dataset & Correlated Video Stream
    base_ds = build_dataset(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        corruption=corr,
        severity=args.severity,
        download=True,
    )

    stream_len = min(args.stream_length, args.max_samples) if args.max_samples else args.stream_length
    stream_ds = CorrelatedStreamDataset(
        base_dataset=base_ds,
        stream_length=stream_len,
        repeat_prob=args.repeat_prob,
        noise_std=args.noise_std,
        jitter_shift=args.jitter_shift,
        seed=args.seed,
    )

    # 2. Build Backbone and TTA Wrapper
    num_classes = 10 if "cifar" in args.dataset.lower() else 1000
    backbone = build_backbone(
        name=args.backbone,
        num_classes=num_classes,
        pretrained=args.pretrained,
        dataset=args.dataset,
    ).to(device)

    if args.model == "source":
        model = Source(backbone).to(device)
    elif args.model == "tent":
        model = Tent(backbone, lr=args.lr).to(device)
    elif args.model == "eata":
        model = Eata(backbone, lr=args.lr, num_classes=num_classes).to(device)
    elif args.model == "sar":
        model = Sar(backbone, lr=args.lr, num_classes=num_classes).to(device)
    elif args.model == "cotta":
        model = Cotta(backbone, lr=args.lr).to(device)
    elif args.model == "dua":
        model = Dua(backbone, num_classes=num_classes).to(device)
    elif args.model == "memo":
        model = Memo(backbone, lr=args.lr).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # 3. Build Cache Gate (if enabled)
    use_gating = args.gating.lower() != "none"
    gate = None
    if use_gating:
        gate = CacheGate(
            metric_name=args.gating,
            threshold=args.threshold,
            capacity=args.cache_capacity,
            policy=args.cache_policy,
            auto_update=True,
        )

    # 4. Initialize Tracker and Logger
    tracker = TTAMetricsTracker()
    logger = ExperimentLogger(
        exp_name=exp_name,
        log_dir=args.log_dir,
        use_tensorboard=args.tensorboard,
        use_wandb=args.wandb,
        config=vars(args),
    )

    # 5. Continuous Streaming Adaptation Loop
    print("\nStarting continuous test-time stream execution...\n")
    for t in range(len(stream_ds)):
        frame, label, meta = stream_ds[t]
        frame = frame.unsqueeze(0).to(device)  # (1, C, H, W)
        target = torch.tensor([label], device=device)

        # Step 5a: Gating decision
        gating_time_ms = 0.0
        should_adapt = True
        sim_val = 0.0
        cache_hit = False

        if use_gating and gate is not None:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_gate_start = time.perf_counter()

            decision = gate.decide(frame)

            if device.type == "cuda":
                torch.cuda.synchronize()
            gating_time_ms = (time.perf_counter() - t_gate_start) * 1000.0

            should_adapt = decision.should_adapt
            sim_val = decision.similarity
            cache_hit = decision.cache_hit

        # Step 5b: Model Forward and/or Adaptation
        forward_time_ms = 0.0
        backward_time_ms = 0.0

        if should_adapt and args.model != "source":
            # Gated Adaptation: compute forward + loss + backward + optimizer step
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_adapt_start = time.perf_counter()

            logits, loss_dict = model.adapt_and_forward(frame)

            if device.type == "cuda":
                torch.cuda.synchronize()
            backward_time_ms = (time.perf_counter() - t_adapt_start) * 1000.0
            adapted = True
        else:
            # Gated Inference: pure forward pass without gradient computation
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_fwd_start = time.perf_counter()

            logits = model.forward(frame)

            if device.type == "cuda":
                torch.cuda.synchronize()
            forward_time_ms = (time.perf_counter() - t_fwd_start) * 1000.0
            adapted = False

        # Step 5c: Update Metrics & Logging
        tracker.update(
            predicted=logits,
            target=target,
            adapted=adapted,
            gating_time_ms=gating_time_ms,
            forward_time_ms=forward_time_ms,
            backward_time_ms=backward_time_ms,
            cache_hit=cache_hit,
        )

        step_metrics = {
            "accuracy": tracker.accuracy,
            "rolling_accuracy": tracker.rolling_accuracy,
            "update_ratio": tracker.update_ratio,
            "similarity": sim_val,
            "avg_total_ms": tracker.avg_total_latency,
        }
        logger.log_step(t + 1, step_metrics)

        if (t + 1) % args.print_freq == 0 or (t + 1) == len(stream_ds):
            logger.print_progress(t + 1, len(stream_ds), step_metrics)

    # 6. Final Results Reporting
    summary = tracker.summary()
    logger.print_summary_table(summary)
    logger.close()


if __name__ == "__main__":
    main()
