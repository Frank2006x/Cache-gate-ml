"""
Online metrics, latency profiling, update ratio, and FLOP trackers for TTA.
"""

from typing import Dict, List, Optional
import time
import torch


class TTAMetricsTracker:
    """
    Comprehensive online metrics tracker for continuous test-time adaptation.

    Tracks:
    - Streaming top-1 accuracy (cumulative and rolling window)
    - Adaptation / update ratio (percentage of frames triggering backprop)
    - Fine-grained latency breakdown: gating, forward inference, backward adaptation
    - Theoretical compute (FLOPs) savings
    """

    def __init__(self, window_size: int = 100, base_forward_mflops: float = 40.0):
        self.window_size = window_size
        self.base_forward_mflops = base_forward_mflops
        # Backward pass roughly costs 2x forward pass for updated layers,
        # or for BN-only TTA, backward overhead of gradient graph + BN backward + optimizer step
        self.backward_multiplier = 2.0

        self.reset()

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.total_frames = 0
        self.total_correct = 0
        self.total_adaptations = 0
        self.total_cache_hits = 0

        # Latency lists in milliseconds
        self.gating_latencies: List[float] = []
        self.forward_latencies: List[float] = []
        self.backward_latencies: List[float] = []
        self.total_latencies: List[float] = []

        # Rolling accuracy window
        self.recent_correct: List[int] = []

    def update(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        adapted: bool,
        gating_time_ms: float = 0.0,
        forward_time_ms: float = 0.0,
        backward_time_ms: float = 0.0,
        cache_hit: bool = False,
    ) -> None:
        """
        Record result and timing for a single frame or batch.
        """
        if predicted.dim() > 1 and predicted.shape[-1] > 1:
            preds = predicted.argmax(dim=-1)
        else:
            preds = predicted

        if target.dim() > 0 and preds.shape != target.shape:
            target = target.view_as(preds)

        correct = int((preds == target).sum().item())
        batch_size = preds.size(0) if preds.dim() > 0 else 1

        self.total_frames += batch_size
        self.total_correct += correct

        self.recent_correct.extend([int((p == t).item()) for p, t in zip(preds.view(-1), target.view(-1))])
        if len(self.recent_correct) > self.window_size:
            self.recent_correct = self.recent_correct[-self.window_size:]

        if adapted:
            self.total_adaptations += 1
        if cache_hit:
            self.total_cache_hits += 1

        self.gating_latencies.append(gating_time_ms)
        self.forward_latencies.append(forward_time_ms)
        self.backward_latencies.append(backward_time_ms)
        self.total_latencies.append(gating_time_ms + forward_time_ms + backward_time_ms)

    @property
    def accuracy(self) -> float:
        """Cumulative top-1 accuracy in percentage [0.0, 100.0]."""
        return (self.total_correct / self.total_frames * 100.0) if self.total_frames > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Cumulative top-1 error rate in percentage [0.0, 100.0]."""
        return 100.0 - self.accuracy

    @property
    def rolling_accuracy(self) -> float:
        """Rolling window top-1 accuracy in percentage."""
        return (sum(self.recent_correct) / len(self.recent_correct) * 100.0) if self.recent_correct else 0.0

    @property
    def update_ratio(self) -> float:
        """Fraction of frames that triggered parameter adaptation."""
        return (self.total_adaptations / self.total_frames) if self.total_frames > 0 else 0.0

    @property
    def cache_hit_ratio(self) -> float:
        """Fraction of frames that were recognized as redundant."""
        return (self.total_cache_hits / self.total_frames) if self.total_frames > 0 else 0.0

    @property
    def avg_gating_latency(self) -> float:
        """Average gating decision latency in milliseconds."""
        return (sum(self.gating_latencies) / len(self.gating_latencies)) if self.gating_latencies else 0.0

    @property
    def avg_forward_latency(self) -> float:
        """Average forward inference latency in milliseconds."""
        return (sum(self.forward_latencies) / len(self.forward_latencies)) if self.forward_latencies else 0.0

    @property
    def avg_backward_latency(self) -> float:
        """Average backward adaptation latency in milliseconds."""
        return (sum(self.backward_latencies) / len(self.backward_latencies)) if self.backward_latencies else 0.0

    @property
    def avg_total_latency(self) -> float:
        """Average total end-to-end latency per frame in milliseconds."""
        return (sum(self.total_latencies) / len(self.total_latencies)) if self.total_latencies else 0.0

    def compute_theoretical_compute_savings(self) -> Dict[str, float]:
        """
        Compute theoretical FLOP reduction compared to full continuous adaptation.
        Baseline: 1 Forward + 1 Backward every frame = (1 + backward_multiplier) * F
        Gated: 1 Forward every frame + UpdateRatio * Backward = (1 + UpdateRatio * backward_multiplier) * F
        """
        f = self.base_forward_mflops
        b = f * self.backward_multiplier
        baseline_cost = (f + b) * self.total_frames
        gated_cost = (f * self.total_frames) + (b * self.total_adaptations)

        flops_saved = max(0.0, baseline_cost - gated_cost)
        pct_savings = (flops_saved / baseline_cost * 100.0) if baseline_cost > 0 else 0.0

        return {
            "baseline_mflops": baseline_cost,
            "gated_mflops": gated_cost,
            "flops_saved_mflops": flops_saved,
            "savings_percent": pct_savings,
        }

    def summary(self) -> Dict[str, float]:
        """Return full summary dictionary of performance, speed, and efficiency."""
        savings = self.compute_theoretical_compute_savings()
        return {
            "total_frames": float(self.total_frames),
            "accuracy": self.accuracy,
            "error_rate": self.error_rate,
            "rolling_accuracy": self.rolling_accuracy,
            "update_ratio": self.update_ratio,
            "cache_hit_ratio": self.cache_hit_ratio,
            "avg_gating_ms": self.avg_gating_latency,
            "avg_forward_ms": self.avg_forward_latency,
            "avg_backward_ms": self.avg_backward_latency,
            "avg_total_ms": self.avg_total_latency,
            "compute_savings_pct": savings["savings_percent"],
        }
