"""Live training observatory for PSC CodonOptimizer.

The observatory is deliberately telemetry-only: it consumes tensors and metrics
already produced by the training step and does not perform an extra forward pass.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from chimera.codon_optimizer import ALL_CODONS, CODON_TO_IDX

try:
    import psutil
except ImportError:  # optional dependency
    psutil = None


@dataclass
class TrainingTelemetry:
    epoch: int
    total_epochs: int
    global_step: int
    total_steps: int
    loss: float
    ce: float
    cai: float
    gc: float
    motif: float
    upa: float
    expression_loss: float
    gradient_norm: float
    learning_rate: float
    cpu_percent: Optional[float]
    memory_gb: Optional[float]
    batch_time: float
    samples_per_second: float
    protein_length: int
    codon_length: int
    predicted_expression: float
    target_expression: float
    architecture: str
    sample_protein: str
    sample_probabilities: Sequence[float]
    sample_candidate_codons: Sequence[str]
    epoch_history: Optional[dict[str, list[float]]] = None


def read_system_stats() -> tuple[Optional[float], Optional[float]]:
    if psutil is None:
        return None, None
    return float(psutil.cpu_percent(interval=None)), float(psutil.virtual_memory().used / (1024**3))


class CodonObservatory:
    """Matplotlib dashboard updated from training telemetry."""

    def __init__(self, update_every: int = 5, history_size: int = 5000) -> None:
        self.update_every = max(1, int(update_every))
        self.history_size = max(100, int(history_size))
        self.step_history: deque[int] = deque(maxlen=self.history_size)
        self.loss_history: deque[float] = deque(maxlen=self.history_size)
        self.ce_history: deque[float] = deque(maxlen=self.history_size)
        self.cai_history: deque[float] = deque(maxlen=self.history_size)
        self.gc_history: deque[float] = deque(maxlen=self.history_size)
        self.motif_history: deque[float] = deque(maxlen=self.history_size)
        self.expr_history: deque[float] = deque(maxlen=self.history_size)
        self.grad_history: deque[float] = deque(maxlen=self.history_size)
        self.lr_history: deque[float] = deque(maxlen=self.history_size)
        self._last_render = 0
        self._started = time.perf_counter()

        plt.ion()
        self.fig = plt.figure(figsize=(16, 10), constrained_layout=True)
        gs = self.fig.add_gridspec(4, 4)
        self.ax_loss = self.fig.add_subplot(gs[0, :2])
        self.ax_obj = self.fig.add_subplot(gs[0, 2:])
        self.ax_bio = self.fig.add_subplot(gs[1, :2])
        self.ax_probs = self.fig.add_subplot(gs[1, 2:])
        self.ax_internals = self.fig.add_subplot(gs[2:, :2])
        self.ax_runtime = self.fig.add_subplot(gs[2:, 2:])
        self.fig.suptitle("PSC CodonOptimizer Training Live Visualizer", fontsize=16)
        self.fig.show()

    def _clear(self, ax, title: str) -> None:
        ax.clear()
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    def update(self, telemetry: TrainingTelemetry) -> None:
        self.step_history.append(telemetry.global_step)
        self.loss_history.append(telemetry.loss)
        self.ce_history.append(telemetry.ce)
        self.cai_history.append(telemetry.cai)
        self.gc_history.append(telemetry.gc)
        self.motif_history.append(telemetry.motif)
        self.expr_history.append(telemetry.expression_loss)
        self.grad_history.append(telemetry.gradient_norm)
        self.lr_history.append(telemetry.learning_rate)

        if telemetry.global_step - self._last_render < self.update_every:
            return
        self._last_render = telemetry.global_step

        x = np.asarray(self.step_history)
        self._clear(self.ax_loss, "Loss")
        self.ax_loss.plot(x, self.loss_history, label="total")
        self.ax_loss.plot(x, self.ce_history, label="CE")
        self.ax_loss.set_xlabel("global step")
        self.ax_loss.legend(loc="upper right")

        self._clear(self.ax_obj, "Objective components")
        self.ax_obj.plot(x, self.cai_history, label="CAI")
        self.ax_obj.plot(x, self.gc_history, label="GC")
        self.ax_obj.plot(x, self.motif_history, label="motif")
        self.ax_obj.plot(x, self.expr_history, label="expression")
        self.ax_obj.set_xlabel("global step")
        self.ax_obj.legend(loc="upper right")

        self._clear(self.ax_bio, "Current biological / optimization metrics")
        labels = ["CAI", "GC", "Motif", "Expr loss", "Grad"]
        values = [telemetry.cai, telemetry.gc, telemetry.motif, telemetry.expression_loss, telemetry.gradient_norm]
        self.ax_bio.bar(labels, values)
        self.ax_bio.tick_params(axis="x", rotation=20)

        self._clear(self.ax_probs, "Synonymous codon probabilities")
        codons = list(telemetry.sample_candidate_codons)
        probs = list(telemetry.sample_probabilities)
        if codons and probs:
            order = np.argsort(probs)[::-1]
            order = order[: min(8, len(order))]
            self.ax_probs.bar(np.arange(len(order)), np.asarray(probs)[order])
            self.ax_probs.set_xticks(np.arange(len(order)))
            self.ax_probs.set_xticklabels([codons[i] for i in order], rotation=45, ha="right")
            self.ax_probs.set_ylim(0, max(1e-6, float(max(probs)) * 1.15))
            self.ax_probs.set_ylabel("P(codon | context)")
        else:
            self.ax_probs.text(0.5, 0.5, "No codon telemetry", ha="center", va="center")

        self._clear(self.ax_internals, "Model internals")
        self.ax_internals.axis("off")
        text = (
            "Protein\n"
            "   ↓\n"
            "ESM-2 (frozen)\n"
            "   ↓\n"
            "Projection\n"
            "   ↓\n"
            "Causal TransformerDecoder\n"
            "   ↓\n"
            "Codon logits [64]\n"
            "   ↓\n"
            "Hard synonymous mask\n"
            "   ↓\n"
            "Predicted DNA\n\n"
            f"sample protein length : {telemetry.protein_length}\n"
            f"sample codon length   : {telemetry.codon_length}\n"
            f"sample protein       : {telemetry.sample_protein[:60]}"
        )
        self.ax_internals.text(0.02, 0.98, text, va="top", family="monospace", fontsize=10)

        self._clear(self.ax_runtime, "Runtime / optimization")
        self.ax_runtime.axis("off")
        elapsed = max(1e-6, time.perf_counter() - self._started)
        text = (
            f"epoch             : {telemetry.epoch}/{telemetry.total_epochs}\n"
            f"global step       : {telemetry.global_step}/{telemetry.total_steps}\n"
            f"learning rate     : {telemetry.learning_rate:.3e}\n"
            f"gradient norm     : {telemetry.gradient_norm:.4f}\n"
            f"batch time        : {telemetry.batch_time:.3f} s\n"
            f"samples/sec      : {telemetry.samples_per_second:.2f}\n"
            f"CPU               : {telemetry.cpu_percent if telemetry.cpu_percent is not None else 'n/a'}%\n"
            f"RAM used          : {telemetry.memory_gb if telemetry.memory_gb is not None else 'n/a'} GB\n"
            f"pred expression   : {telemetry.predicted_expression:.4f}\n"
            f"target expression : {telemetry.target_expression:.4f}\n"
            f"uptime            : {elapsed:.1f} s\n\n"
            f"architecture\n{telemetry.architecture}"
        )
        self.ax_runtime.text(0.02, 0.98, text, va="top", family="monospace", fontsize=10)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def close(self) -> None:
        plt.ioff()
        plt.close(self.fig)


def candidate_codons_for_amino_acid(amino_acid: str) -> list[str]:
    """Return the exact synonymous codon set used by the model."""
    from chimera.codon_optimizer import CODON_TABLE
    return list(CODON_TABLE[amino_acid])
