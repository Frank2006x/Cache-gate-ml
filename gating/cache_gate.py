"""
Core CacheGate implementation for Test-Time Adaptation.
Decoupled gating module that decides whether to trigger expensive gradient adaptation
based on perceptual similarity to cached reference frames.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
import time
import torch
from .similarity import get_similarity_metric


@dataclass
class GateDecision:
    """Structured decision output from CacheGate."""
    should_adapt: bool
    similarity: float
    cache_hit: bool
    is_warmup: bool
    gating_latency_ms: float
    info: Dict[str, Union[float, int, str]] = field(default_factory=dict)


class CacheGate:
    """
    Decoupled algorithmic cache gate for continuous TTA.

    Args:
        metric_name: Similarity metric ('ssim', 'phash', 'l1', 'l2', 'cosine').
        threshold: Similarity threshold in [0, 1]. If similarity >= threshold,
                   the frame is considered redundant (cache hit), skipping adaptation.
        capacity: Number of reference frames to retain in cache (default: 1 keyframe).
        policy: Cache replacement policy: 'keyframe' (single keyframe updated on novel frame),
                'fifo' (sliding window buffer of size capacity),
                or 'ema' (exponential moving average frame).
        auto_update: If True, automatically updates cache when should_adapt is True.
        **metric_kwargs: Additional arguments forwarded to the similarity metric.
    """

    def __init__(
        self,
        metric_name: str = "ssim",
        threshold: float = 0.85,
        capacity: int = 1,
        policy: str = "keyframe",
        auto_update: bool = True,
        **metric_kwargs,
    ):
        self.metric_name = metric_name
        self.threshold = float(threshold)
        self.capacity = max(1, capacity)
        self.policy = policy.lower().strip()
        self.auto_update = auto_update
        self.metric_fn = get_similarity_metric(metric_name, **metric_kwargs)

        # Internal cache state
        self.cache_buffer: List[torch.Tensor] = []
        self.total_queries = 0
        self.total_hits = 0
        self.total_updates = 0

    def reset(self) -> None:
        """Clear all cached frames and reset running counters."""
        self.cache_buffer.clear()
        self.total_queries = 0
        self.total_hits = 0
        self.total_updates = 0

    def update_cache(self, x: torch.Tensor) -> None:
        """
        Store frame into cache according to policy.
        Frame is detached and stored without computational graph.
        """
        frame = x.detach().clone()
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)

        if self.policy == "keyframe":
            # Overwrite single keyframe
            self.cache_buffer = [frame]
        elif self.policy == "fifo":
            if len(self.cache_buffer) >= self.capacity:
                self.cache_buffer.pop(0)
            self.cache_buffer.append(frame)
        elif self.policy == "ema":
            if not self.cache_buffer:
                self.cache_buffer = [frame]
            else:
                alpha = 0.7  # Smoothing factor
                self.cache_buffer[0] = alpha * self.cache_buffer[0] + (1.0 - alpha) * frame
        else:
            raise ValueError(f"Unknown cache policy: {self.policy}")

        self.total_updates += 1

    def decide(self, x: torch.Tensor) -> GateDecision:
        """
        Evaluate incoming frame against cache and return adaptation decision.

        Args:
            x: Input frame tensor (B, C, H, W) or (C, H, W).

        Returns:
            GateDecision containing should_adapt flag, similarity score, and diagnostic info.
        """
        start_time = time.perf_counter()
        self.total_queries += 1

        # Frame 0 / Cold cache warmup: cache is empty, must adapt and initialize cache
        if not self.cache_buffer:
            if self.auto_update:
                self.update_cache(x)

            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return GateDecision(
                should_adapt=True,
                similarity=0.0,
                cache_hit=False,
                is_warmup=True,
                gating_latency_ms=latency_ms,
                info={"policy": self.policy, "threshold": self.threshold, "metric": self.metric_name},
            )

        # Compute similarity against cached frame(s)
        x_in = x.detach()
        if x_in.dim() == 3:
            x_in = x_in.unsqueeze(0)

        similarities = []
        with torch.no_grad():
            for cached_frame in self.cache_buffer:
                sim = self.metric_fn(x_in, cached_frame)
                if isinstance(sim, torch.Tensor):
                    sim = sim.item()
                similarities.append(float(sim))

        max_sim = max(similarities) if similarities else 0.0

        # Decision rule:
        # If similarity >= threshold: frame is visually redundant -> CACHE HIT -> SKIP GRADIENT UPDATE
        # If similarity < threshold: visual shift detected -> CACHE MISS -> RUN GRADIENT UPDATE
        cache_hit = max_sim >= self.threshold
        should_adapt = not cache_hit

        if cache_hit:
            self.total_hits += 1
        else:
            if self.auto_update:
                self.update_cache(x)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return GateDecision(
            should_adapt=should_adapt,
            similarity=max_sim,
            cache_hit=cache_hit,
            is_warmup=False,
            gating_latency_ms=latency_ms,
            info={
                "policy": self.policy,
                "threshold": self.threshold,
                "metric": self.metric_name,
                "hit_ratio": self.hit_ratio,
            },
        )

    @property
    def hit_ratio(self) -> float:
        """Fraction of incoming queries that hit the cache (avoided adaptation)."""
        return (self.total_hits / self.total_queries) if self.total_queries > 0 else 0.0

    @property
    def update_ratio(self) -> float:
        """Fraction of incoming queries that triggered an adaptation step."""
        return (self.total_updates / self.total_queries) if self.total_queries > 0 else 0.0
