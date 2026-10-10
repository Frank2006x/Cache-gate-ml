"""
Unified logger supporting formatted console tables, TensorBoard, and Weights & Biases.
"""

from typing import Any, Dict, Optional
from pathlib import Path
import json
import sys


class ExperimentLogger:
    """
    Handles experiment logging across multiple backends.
    """

    def __init__(
        self,
        exp_name: str = "tta_experiment",
        log_dir: str = "./runs",
        use_tensorboard: bool = False,
        use_wandb: bool = False,
        wandb_project: str = "cache_gated_tta",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.exp_name = exp_name
        self.log_dir = Path(log_dir) / exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb

        # Save run configuration
        if config:
            with open(self.log_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)

        # TensorBoard setup
        self.tb_writer = None
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                print("[Logger] Warning: TensorBoard not installed. Skipping TB logging.")

        # WandB setup
        if self.use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, name=exp_name, config=config)
            except ImportError:
                print("[Logger] Warning: wandb not installed. Skipping WandB logging.")
                self.use_wandb = False

    def log_step(self, step: int, metrics: Dict[str, float]) -> None:
        """Log per-step metrics to backends."""
        if self.tb_writer:
            for k, v in metrics.items():
                self.tb_writer.add_scalar(k, v, step)

        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)

    def print_progress(self, step: int, total: int, metrics: Dict[str, float]) -> None:
        """Print clean single-line progress update."""
        acc = metrics.get("accuracy", 0.0)
        roll_acc = metrics.get("rolling_accuracy", 0.0)
        up_ratio = metrics.get("update_ratio", 0.0)
        sim = metrics.get("similarity", 0.0)
        total_lat = metrics.get("avg_total_ms", 0.0)

        sys.stdout.write(
            f"\r[{step:04d}/{total:04d}] "
            f"Acc: {acc:5.2f}% | RollAcc: {roll_acc:5.2f}% | "
            f"UpdateRatio: {up_ratio*100:5.1f}% | Sim: {sim:.3f} | Latency: {total_lat:4.1f}ms"
        )
        sys.stdout.flush()

    def print_summary_table(self, summary: Dict[str, Any]) -> None:
        """Print final formatted summary table."""
        print("\n" + "=" * 65)
        print(f"             EXPERIMENT SUMMARY: {self.exp_name}")
        print("=" * 65)
        print(f"  Total Processed Frames    : {int(summary.get('total_frames', 0))}")
        print(f"  Top-1 Accuracy            : {summary.get('accuracy', 0.0):.2f} %")
        print(f"  Top-1 Error Rate          : {summary.get('error_rate', 0.0):.2f} %")
        print(f"  Adaptation Update Ratio   : {summary.get('update_ratio', 0.0)*100.0:.2f} % (Adaptations: {int(summary.get('total_frames', 0) * summary.get('update_ratio', 0.0))})")
        print(f"  Cache Hit Ratio           : {summary.get('cache_hit_ratio', 0.0)*100.0:.2f} %")
        print(f"  Theoretical Compute Saved : {summary.get('compute_savings_pct', 0.0):.2f} % FLOPs")
        print("-" * 65)
        print("  Latency Breakdown (Per Frame):")
        print(f"    - Gating Decision       : {summary.get('avg_gating_ms', 0.0):.3f} ms")
        print(f"    - Forward Inference     : {summary.get('avg_forward_ms', 0.0):.3f} ms")
        print(f"    - Backward / Adaptation : {summary.get('avg_backward_ms', 0.0):.3f} ms")
        print(f"    - Total End-to-End      : {summary.get('avg_total_ms', 0.0):.3f} ms")
        print("=" * 65)

        # Save summary JSON
        with open(self.log_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    def close(self) -> None:
        """Close loggers."""
        if self.tb_writer:
            self.tb_writer.close()
        if self.use_wandb:
            import wandb
            wandb.finish()
