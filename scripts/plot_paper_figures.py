"""
Generate publication-quality figures for the conference paper:
1. Figure 1: Latency & Speedup Pareto Frontier across Cache Hit Ratios.
2. Figure 2: Model Scaling Law (ResNet-18 vs ResNet-50 Backprop vs SSIM Gate Overhead).
3. Figure 3: Multi-Corruption Latency Breakdown on CUDA.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

out_dir = Path("./figures")
out_dir.mkdir(parents=True, exist_ok=True)


def plot_pareto_frontier():
    """Plot latency speedup as a function of stream redundancy."""
    hit_ratios = np.linspace(0.0, 1.0, 100)

    # Measured empirical values on CUDA ResNet-18 (Fog benchmark)
    t_gate = 1.064
    t_fwd = 2.266
    t_bwd = 4.564
    t_baseline = t_fwd + t_bwd

    t_gated = t_gate + t_fwd + (1.0 - hit_ratios) * t_bwd
    speedup = t_baseline / t_gated

    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))

    color1 = "#1f77b4"
    ax1.set_xlabel("Stream Redundancy / Cache Hit Ratio (%)", fontweight="bold")
    ax1.set_ylabel("Per-Frame Latency (ms)", color=color1, fontweight="bold")
    ax1.plot(hit_ratios * 100, [t_baseline] * len(hit_ratios), "r--", linewidth=2, label="Dense Adaptation (No Gating)")
    ax1.plot(hit_ratios * 100, t_gated, color=color1, linewidth=2.5, label="Cache-Gated TTA (Ours)")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 8)

    # Breakeven point
    r_be = (t_gate / t_bwd) * 100
    ax1.axvline(x=r_be, color="gray", linestyle=":", linewidth=1.5, label=f"Breakeven Point ({r_be:.1f}%)")

    ax2 = ax1.twinx()
    color2 = "#2ca02c"
    ax2.set_ylabel("Wall-Clock Speedup Factor (×)", color=color2, fontweight="bold")
    ax2.plot(hit_ratios * 100, speedup, color=color2, linewidth=2.5, linestyle="-.", label="Speedup Factor")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0.8, 2.5)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", frameon=True)

    plt.title("Empirical Latency Reduction & Speedup on CUDA", fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_dir / "fig1_pareto_speedup.pdf")
    plt.savefig(out_dir / "fig1_pareto_speedup.png")
    plt.close()
    print("Saved Figure 1 to figures/fig1_pareto_speedup.png")


def plot_scaling_law():
    """Plot the model scaling law: constant gate overhead vs ballooning backprop cost."""
    models = ["ResNet-18\n(TENT)", "ResNet-18\n(CoTTA)", "ResNet-50\n(TENT)", "ResNet-50\n(CoTTA)"]
    gate_costs = [3.09, 3.09, 1.69, 1.69]
    bwd_costs = [15.75, 51.39, 44.83, 126.68]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.2))
    rects1 = ax.bar(x - width/2, gate_costs, width, label="SSIM Gate Overhead", color="#4e79a7")
    rects2 = ax.bar(x + width/2, bwd_costs, width, label="Adaptation Backprop Cost", color="#e15759")

    ax.set_ylabel("Wall-Clock Time (ms)", fontweight="bold")
    ax.set_title("Model Scaling Law: $O(1)$ Gating vs. $O(N)$ Backprop Cost", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate speedup ratios
    ratios = [bwd_costs[i] / gate_costs[i] for i in range(len(models))]
    for i, r in enumerate(ratios):
        ax.text(x[i] + width/2, bwd_costs[i] + 3, f"{r:.1f}×\nSlower", ha="center", fontsize=9, fontweight="bold", color="#a83232")

    plt.tight_layout()
    plt.savefig(out_dir / "fig2_scaling_law.pdf")
    plt.savefig(out_dir / "fig2_scaling_law.png")
    plt.close()
    print("Saved Figure 2 to figures/fig2_scaling_law.png")


def plot_corruption_throughput():
    """Plot CUDA throughput (FPS) across corruption benchmarks."""
    corruptions = ["Gaussian Noise\n(Severity 5)", "Defocus Blur\n(Severity 5)", "Fog\n(Severity 5)"]
    baseline_fps = [1000.0 / 25.34, 1000.0 / 18.90, 1000.0 / 13.50]
    gated_fps = [1000.0 / 15.09, 1000.0 / 11.90, 1000.0 / 7.89]

    x = np.arange(len(corruptions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    rects1 = ax.bar(x - width/2, baseline_fps, width, label="Baseline (Dense Adaptation)", color="#79706e")
    rects2 = ax.bar(x + width/2, gated_fps, width, label="Cache-Gated TTA (Ours)", color="#59a14f")

    ax.set_ylabel("Throughput (Frames Per Second)", fontweight="bold")
    ax.set_title("Streaming Throughput on CUDA (ResNet-18)", fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(corruptions)
    ax.axhline(y=60, color="orange", linestyle="--", label="Real-Time 60 FPS Target")
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for i in range(len(corruptions)):
        ax.text(x[i] + width/2, gated_fps[i] + 2, f"{gated_fps[i]:.1f}\nFPS", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_dir / "fig3_throughput_fps.pdf")
    plt.savefig(out_dir / "fig3_throughput_fps.png")
    plt.close()
    print("Saved Figure 3 to figures/fig3_throughput_fps.png")


if __name__ == "__main__":
    plot_pareto_frontier()
    plot_scaling_law()
    plot_corruption_throughput()
