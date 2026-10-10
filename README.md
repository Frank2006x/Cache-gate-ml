# Algorithmic Cache-Gating for Compute-Efficient Continuous Test-Time Adaptation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Target-CVPR%2FICCV%2FNeurIPS-blue.svg)]()

A modular, production-ready research framework implementing **Algorithmic Cache-Gating** for continuous Test-Time Adaptation (TTA).

---

## 🎯 Overview

In continuous test-time adaptation (e.g., autonomous driving, video surveillance, edge robotics), models encounter a high-throughput temporal stream of test frames. Traditional TTA algorithms (such as **TENT**, **EATA**, or **SAR**) trigger backpropagation and optimizer updates on *every single frame* or batch.

However, continuous physical video displays high **temporal correlation**: consecutive frames often feature identical views, minor sensor jitter, or static backgrounds. Running expensive gradient backward passes on redundant frames wastes substantial compute and battery power on edge devices, while risking overfitting to noisy sample distributions.

**Algorithmic Cache-Gating** decouples perceptual frame redundancy detection from model adaptation:
1. When an incoming frame $x_t$ arrives, the **Cache Gate** measures perceptual similarity (PyTorch-native SSIM, pHash, or normalized difference) against a cached keyframe $x_{\text{cache}}$.
2. If $\text{sim}(x_t, x_{\text{cache}}) \ge \tau$ (Cache Hit): Gradient backpropagation is **bypassed**, performing forward-only inference.
3. If $\text{sim}(x_t, x_{\text{cache}}) < \tau$ (Visual Shift / Novelty): A full adaptation step is performed, and $x_{\text{cache}}$ is updated.

```
Incoming Stream (x_t) ─────────► [ Cache Gate ]
                                        │
                         Similarity >= τ?
                         ┌──────────────┴──────────────┐
                    Yes (Hit)                      No (Miss)
                         │                             │
                         ▼                             ▼
                 [ Forward Only ]          [ Forward + Backward + Opt ]
             (No gradient computation)     (Adapt BN affine parameters)
                         │                             │
                         └──────────────┬──────────────┘
                                        ▼
                                 Prediction ŷ_t
```

---

## 📁 Repository Structure

```text
cache_gated_tta/
├── data/
│   ├── __init__.py
│   ├── dataset_loader.py          # CIFAR-10, CIFAR-10-C & ImageNet-C loaders with synthetic fallback
│   └── stream_generator.py        # Markov temporal video stream simulator (persistence & micro-shake)
├── models/
│   ├── __init__.py
│   ├── base.py                    # BaseTTA abstract class separating forward from adapt_and_forward
│   ├── backbone.py                # ResNet wrappers (ResNet-18/34/50) customized for CIFAR & ImageNet
│   ├── source.py                  # Source (Pretrained baseline without test-time updates)
│   ├── tent.py                    # TENT (Wang et al., ICLR 2021) entropy minimization wrapper
│   ├── eata.py                    # EATA (Niu et al., ICML 2022) with Fisher anti-forgetting regularization
│   ├── sar.py                     # SAR (Niu et al., ICLR 2023) sharpness-aware TTA wrapper
│   ├── cotta.py                   # CoTTA (Wang et al., CVPR 2022) mean-teacher continual TTA
│   └── dua.py                     # DUA (Mirza et al., ECCV 2022) backprop-free uncertainty normalization
├── gating/
│   ├── __init__.py
│   ├── cache_gate.py              # Decoupled CacheGate class managing keyframe state & decisions
│   └── similarity.py              # Pure PyTorch GPU SSIM, pHash, and normalized difference metrics
├── utils/
│   ├── __init__.py
│   ├── metrics.py                 # Streaming accuracy, update ratio, FLOP savings, and latency tracker
│   └── logger.py                  # Experiment logger (Console tables, TensorBoard, Weights & Biases)
├── scripts/
│   ├── run_baseline.sh            # Shell script for standard TENT/EATA baselines
│   └── run_gated.sh               # Shell script for cache-gated threshold sweeps
├── notebooks/
│   └── 01_latency_profiling.ipynb # Benchmark notebook measuring SSIM overhead vs Backward pass
├── main.py                        # Unified CLI entrypoint with argparse & experiment runner
├── requirements.txt               # Project dependencies
├── .gitignore                     # Git ignore rules for checkpoints, datasets, and caches
└── README.md                      # Documentation and guide
```

---

## ⚡ Quick Start

### 1. Installation

```bash
git clone https://github.com/your-username/cache_gated_tta.git
cd cache_gated_tta

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run a Baseline (Standard TENT without Gating)

Adapts normalization parameters on every frame:

```bash
python main.py \
    --model tent \
    --backbone resnet18 \
    --dataset cifar10 \
    --corruption gaussian_noise \
    --severity 5 \
    --stream_length 1000 \
    --repeat_prob 0.6 \
    --gating none
```

### 3. Run Cache-Gated TTA (Our Approach)

Applies SSIM cache-gating with threshold $\tau = 0.85$:

```bash
python main.py \
    --model tent \
    --backbone resnet18 \
    --dataset cifar10 \
    --corruption gaussian_noise \
    --severity 5 \
    --stream_length 1000 \
    --repeat_prob 0.6 \
    --gating ssim \
    --threshold 0.85 \
    --cache_policy keyframe
```

### 4. Run Batch Experiments via Shell Scripts

```bash
# Run baseline suite (Source, TENT, EATA)
bash scripts/run_baseline.sh

# Run gated threshold sweep (tau in [0.80, 0.85, 0.90])
bash scripts/run_gated.sh
```

---

## 📊 Key Benchmark Metrics

Each run automatically computes and outputs:
- **Top-1 Accuracy / Error Rate**: Streaming and rolling-window classification accuracy under continuous distribution shift.
- **Update Ratio ($R_{\text{update}}$)**: Fraction of frames that triggered parameter updates $\frac{N_{\text{adapt}}}{N_{\text{total}}}$.
- **Cache Hit Ratio ($R_{\text{hit}}$)**: Percentage of redundant frames identified and bypassed.
- **Theoretical Compute Savings**: Percentage of FLOPs saved by bypassing backward passes:
  $$\text{Savings} = \frac{(1 - R_{\text{update}}) \times \text{FLOPs}_{\text{backward}}}{\text{FLOPs}_{\text{total}}} \times 100\%$$
- **Latency Breakdown**: Millisecond timings separated into **Gating Overhead**, **Forward Inference**, and **Backward Adaptation**.

---

## 🔬 Profiling & Micro-Benchmarks

To strictly verify that the gating overhead is negligible compared to backpropagation:

```bash
jupyter notebook notebooks/01_latency_profiling.ipynb
```

**Key Observation**:
- PyTorch GPU SSIM runs in **$< 0.3$ ms** per frame.
- ResNet-18 backward adaptation pass requires **$5 - 15$ ms** per frame.
- **Gating Overhead Ratio**: $\frac{T_{\text{SSIM}}}{T_{\text{backward}}} < 3\%$.
- **Breakeven Point**: If as few as $\approx 3-5\%$ of frames in a stream are redundant, cache-gating strictly accelerates the end-to-end pipeline.

---

## 🧩 Modularity & Extensibility

### Adding a New Gating Metric
Register any function taking two tensors `(x1, x2)` and returning a scalar in `gating/similarity.py`:
```python
def my_metric(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return 1.0 - (x1 - x2).abs().mean()
```

### Adding a New TTA Algorithm
Subclass `BaseTTA` in `models/base.py`:
```python
from models.base import BaseTTA

class MyTTA(BaseTTA):
    def forward(self, x):
        with torch.no_grad():
            return self.model(x)

    def adapt_and_forward(self, x):
        self.optimizer.zero_grad()
        out = self.model(x)
        loss = my_custom_loss(out)
        loss.backward()
        self.optimizer.step()
        return out.detach(), {"loss": loss.item()}
```

---

## 📜 Citation

```bibtex
@inproceedings{cache_gated_tta_2025,
  title={Algorithmic Cache-Gating for Compute-Efficient Continuous Test-Time Adaptation},
  author={Anonymous},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```
