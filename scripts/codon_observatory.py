"""Live training observatory for PSC CodonOptimizer."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from chimera.codon_optimizer import CODON_TABLE

try:
    import psutil
except ImportError:
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

    # Currently inspected position.
    selected_position: int
    selected_amino_acid: str

    # All synonymous candidates at selected position.
    candidate_codons: Sequence[str]
    candidate_probabilities: Sequence[float]

    # Causal teacher-forced prediction.
    predicted_codons: Sequence[str]

    # Cross attention from final decoder layer.
    cross_attention: Optional[np.ndarray] = None


def read_system_stats():

    if psutil is None:
        return None, None

    return (
        float(psutil.cpu_percent(interval=None)),
        float(
            psutil.virtual_memory().used
            / (1024 ** 3)
        ),
    )


class CodonObservatory:

    def __init__(
        self,
        update_every: int = 5,
        history_size: int = 5000,
    ):

        self.update_every = max(
            1,
            int(update_every),
        )

        self.history_size = max(
            100,
            int(history_size),
        )

        self.step_history = deque(
            maxlen=self.history_size
        )

        self.loss_history = deque(
            maxlen=self.history_size
        )

        self.ce_history = deque(
            maxlen=self.history_size
        )

        self.cai_history = deque(
            maxlen=self.history_size
        )

        self.gc_history = deque(
            maxlen=self.history_size
        )

        self.motif_history = deque(
            maxlen=self.history_size
        )

        self.expr_history = deque(
            maxlen=self.history_size
        )

        self.grad_history = deque(
            maxlen=self.history_size
        )

        self._last_render = 0

        self._started = time.perf_counter()

        plt.ion()

        self.fig = plt.figure(
            figsize=(18, 12),
            constrained_layout=True,
        )

        gs = self.fig.add_gridspec(
            4,
            4,
        )

        self.ax_loss = self.fig.add_subplot(
            gs[0, :2]
        )

        self.ax_obj = self.fig.add_subplot(
            gs[0, 2:]
        )

        self.ax_bio = self.fig.add_subplot(
            gs[1, :2]
        )

        self.ax_probs = self.fig.add_subplot(
            gs[1, 2:]
        )

        self.ax_decode = self.fig.add_subplot(
            gs[2, :2]
        )

        self.ax_attention = self.fig.add_subplot(
            gs[2, 2:]
        )

        self.ax_internals = self.fig.add_subplot(
            gs[3, :2]
        )

        self.ax_runtime = self.fig.add_subplot(
            gs[3, 2:]
        )

        self.fig.suptitle(
            "PSC CodonOptimizer Training Live Visualizer",
            fontsize=17,
        )

        self.fig.show()

    def _clear(
        self,
        ax,
        title: str,
    ):

        ax.clear()

        ax.set_title(title)

        ax.grid(
            True,
            alpha=0.25,
        )

    def update(
        self,
        telemetry: TrainingTelemetry,
    ):

        step = telemetry.global_step

        self.step_history.append(step)

        self.loss_history.append(
            telemetry.loss
        )

        self.ce_history.append(
            telemetry.ce
        )

        self.cai_history.append(
            telemetry.cai
        )

        self.gc_history.append(
            telemetry.gc
        )

        self.motif_history.append(
            telemetry.motif
        )

        self.expr_history.append(
            telemetry.expression_loss
        )

        self.grad_history.append(
            telemetry.gradient_norm
        )

        if (
            step - self._last_render
            < self.update_every
        ):
            return

        self._last_render = step

        x = np.asarray(
            self.step_history
        )

        # ==============================================================
        # LOSS
        # ==============================================================

        self._clear(
            self.ax_loss,
            "Loss",
        )

        self.ax_loss.plot(
            x,
            self.loss_history,
            label="total",
        )

        self.ax_loss.plot(
            x,
            self.ce_history,
            label="CE",
        )

        self.ax_loss.set_xlabel(
            "global step"
        )

        self.ax_loss.legend(
            loc="upper right"
        )

        # ==============================================================
        # OBJECTIVES
        # ==============================================================

        self._clear(
            self.ax_obj,
            "Objective components",
        )

        self.ax_obj.plot(
            x,
            self.cai_history,
            label="CAI",
        )

        self.ax_obj.plot(
            x,
            self.gc_history,
            label="GC",
        )

        self.ax_obj.plot(
            x,
            self.motif_history,
            label="motif",
        )

        self.ax_obj.plot(
            x,
            self.expr_history,
            label="expression",
        )

        self.ax_obj.set_xlabel(
            "global step"
        )

        self.ax_obj.legend(
            loc="upper right"
        )

        # ==============================================================
        # BIOLOGICAL METRICS
        # ==============================================================

        self._clear(
            self.ax_bio,
            "Current biological / optimization metrics",
        )

        labels = [
            "CAI",
            "GC",
            "Motif",
            "Expr loss",
            "Grad",
        ]

        values = [
            telemetry.cai,
            telemetry.gc,
            telemetry.motif,
            telemetry.expression_loss,
            telemetry.gradient_norm,
        ]

        self.ax_bio.bar(
            labels,
            values,
        )

        self.ax_bio.tick_params(
            axis="x",
            rotation=20,
        )

        # ==============================================================
        # SYNONYMOUS CODON PROBABILITIES
        # ==============================================================

        self._clear(
            self.ax_probs,
            "Synonymous codon probabilities",
        )

        codons = list(
            telemetry.candidate_codons
        )

        probs = np.asarray(
            telemetry.candidate_probabilities,
            dtype=float,
        )

        if codons and len(probs) == len(codons):

            order = np.argsort(
                probs
            )[::-1]

            self.ax_probs.bar(
                np.arange(len(order)),
                probs[order],
            )

            self.ax_probs.set_xticks(
                np.arange(len(order))
            )

            self.ax_probs.set_xticklabels(
                [
                    codons[i]
                    for i in order
                ],
                rotation=45,
                ha="right",
            )

            self.ax_probs.set_ylim(
                0,
                max(
                    1.0,
                    float(probs.max()) * 1.15,
                ),
            )

            self.ax_probs.set_ylabel(
                "P(codon | context)"
            )

            self.ax_probs.set_title(
                "Synonymous codons for "
                f"{telemetry.selected_amino_acid}"
                f" @ position "
                f"{telemetry.selected_position + 1}"
            )

        # ==============================================================
        # CAUSAL DECODING TRACE
        # ==============================================================

        self._clear(
            self.ax_decode,
            "Causal codon decoding trace",
        )

        predicted = list(
            telemetry.predicted_codons
        )

        protein = telemetry.sample_protein

        if predicted:

            display_limit = min(
                48,
                len(predicted),
            )

            positions = np.arange(
                display_limit
            )

            self.ax_decode.plot(
                positions,
                np.arange(display_limit),
                marker="o",
                linestyle="",
            )

            self.ax_decode.set_xticks(
                positions
            )

            self.ax_decode.set_xticklabels(
                [
                    predicted[i]
                    for i in positions
                ],
                rotation=65,
                ha="right",
                fontsize=7,
            )

            self.ax_decode.set_yticks(
                positions
            )

            self.ax_decode.set_yticklabels(
                [
                    protein[i]
                    for i in positions
                ],
                fontsize=7,
            )

            self.ax_decode.set_xlabel(
                "predicted codon"
            )

            self.ax_decode.set_ylabel(
                "protein residue"
            )

        # ==============================================================
        # CROSS ATTENTION
        # ==============================================================

        self._clear(
            self.ax_attention,
            "Final decoder cross-attention",
        )

        attention = telemetry.cross_attention

        if attention is not None:

            attention = np.asarray(
                attention
            )

            if attention.ndim == 3:

                # heads × target × memory
                attention = attention.mean(
                    axis=0
                )

            if attention.ndim == 2:

                self.ax_attention.imshow(
                    attention,
                    aspect="auto",
                    interpolation="nearest",
                )

                self.ax_attention.set_xlabel(
                    "protein position"
                )

                self.ax_attention.set_ylabel(
                    "decoder codon position"
                )

                self.ax_attention.set_title(
                    "Final-layer protein → codon attention"
                )

            else:

                self.ax_attention.text(
                    0.5,
                    0.5,
                    "Unexpected attention shape",
                    ha="center",
                    va="center",
                )

        else:

            self.ax_attention.text(
                0.5,
                0.5,
                "Attention capture disabled",
                ha="center",
                va="center",
            )

        # ==============================================================
        # MODEL INTERNALS
        # ==============================================================

        self._clear(
            self.ax_internals,
            "Model internals",
        )

        self.ax_internals.axis(
            "off"
        )

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
            f"sample protein length : "
            f"{telemetry.protein_length}\n"
            f"sample codon length   : "
            f"{telemetry.codon_length}\n"
            f"selected position     : "
            f"{telemetry.selected_position + 1}\n"
            f"selected amino acid   : "
            f"{telemetry.selected_amino_acid}\n"
            f"sample protein        : "
            f"{telemetry.sample_protein[:70]}\n"
            f"sample codons         : "
            f"{telemetry.sample_codons[:70]}"
        )

        self.ax_internals.text(
            0.02,
            0.98,
            text,
            va="top",
            family="monospace",
            fontsize=9,
        )

        # ==============================================================
        # RUNTIME
        # ==============================================================

        self._clear(
            self.ax_runtime,
            "Runtime / optimization",
        )

        self.ax_runtime.axis(
            "off"
        )

        elapsed = max(
            1e-6,
            time.perf_counter()
            - self._started,
        )

        text = (
            f"epoch             : "
            f"{telemetry.epoch}/"
            f"{telemetry.total_epochs}\n"
            f"global step       : "
            f"{telemetry.global_step}/"
            f"{telemetry.total_steps}\n"
            f"learning rate     : "
            f"{telemetry.learning_rate:.3e}\n"
            f"gradient norm     : "
            f"{telemetry.gradient_norm:.4f}\n"
            f"batch time        : "
            f"{telemetry.batch_time:.3f} s\n"
            f"samples/sec       : "
            f"{telemetry.samples_per_second:.2f}\n"
            f"CPU               : "
            f"{telemetry.cpu_percent if telemetry.cpu_percent is not None else 'n/a'}%\n"
            f"RAM used          : "
            f"{telemetry.memory_gb if telemetry.memory_gb is not None else 'n/a'} GB\n"
            f"pred expression   : "
            f"{telemetry.predicted_expression:.4f}\n"
            f"target expression : "
            f"{telemetry.target_expression:.4f}\n"
            f"uptime            : "
            f"{elapsed:.1f} s\n\n"
            f"architecture\n"
            f"{telemetry.architecture}"
        )

        self.ax_runtime.text(
            0.02,
            0.98,
            text,
            va="top",
            family="monospace",
            fontsize=9,
        )

        self.fig.canvas.draw_idle()

        self.fig.canvas.flush_events()

        plt.pause(
            0.001
        )

    def close(self):

        plt.ioff()

        plt.close(
            self.fig
        )


def candidate_codons_for_amino_acid(
    amino_acid: str,
) -> list[str]:

    return list(
        CODON_TABLE[amino_acid]
    )