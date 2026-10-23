"""CLI training observatory for PSC CodonOptimizer.

The observatory is intentionally terminal-native. It does not import GUI
or plotting libraries, so enabling training telemetry cannot pull Matplotlib
or another native GUI backend into the training process.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import sys
import time
from typing import Optional, Sequence

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
    selected_position: int
    selected_amino_acid: str
    candidate_codons: Sequence[str]
    candidate_probabilities: Sequence[float]
    predicted_codons: Sequence[str]
    cross_attention: Optional[np.ndarray] = None
    total_parameters: int = 0
    trainable_parameters: int = 0
    model_config: str = "unavailable"


def read_system_stats():
    if psutil is None:
        return None, None
    return (
        float(psutil.cpu_percent(interval=None)),
        float(psutil.virtual_memory().used / (1024**3)),
    )


class CodonObservatory:
    """Compact, terminal-native training telemetry and attention dashboard."""

    _BLOCKS = " .:-=+*#%@"
    _SPARKS = "▁▂▃▄▅▆▇█"

    def __init__(
        self,
        update_every: int = 5,
        history_size: int = 5000,
        max_attention_length: int = 512,
        stream=None,
    ):
        self.update_every = max(1, int(update_every))
        self.history_size = max(100, int(history_size))
        self.max_attention_length = max(1, int(max_attention_length))
        self.stream = stream if stream is not None else sys.stdout

        self.step_history = deque(maxlen=self.history_size)
        self.loss_history = deque(maxlen=self.history_size)
        self.ce_history = deque(maxlen=self.history_size)
        self.cai_history = deque(maxlen=self.history_size)
        self.gc_history = deque(maxlen=self.history_size)
        self.motif_history = deque(maxlen=self.history_size)
        self.upa_history = deque(maxlen=self.history_size)
        self.expr_history = deque(maxlen=self.history_size)
        self.grad_history = deque(maxlen=self.history_size)

        self._last_render = 0
        self._started = time.perf_counter()
        self._closed = False

    def should_capture_attention(
        self,
        global_step: int,
        sequence_length: int,
    ) -> bool:
        return (
            sequence_length <= self.max_attention_length
            and (
                global_step == 0
                or global_step % self.update_every == 0
            )
        )

    @staticmethod
    def _sparkline(values, width: int = 30) -> str:
        values = list(values)
        if not values:
            return ""
        if len(values) > width:
            indices = np.linspace(0, len(values) - 1, width).astype(int)
            values = [values[i] for i in indices]
        finite = [float(v) for v in values if np.isfinite(v)]
        if not finite:
            return "?" * len(values)
        lo, hi = min(finite), max(finite)
        if hi <= lo:
            return "▅" * len(values)
        return "".join(
            CodonObservatory._SPARKS[
                min(
                    len(CodonObservatory._SPARKS) - 1,
                    max(0, int((float(v) - lo) / (hi - lo) * 7)),
                )
            ]
            for v in values
        )

    @staticmethod
    def _bar(value: float, maximum: float = 1.0, width: int = 24) -> str:
        if not np.isfinite(value):
            return "?" * width
        if maximum <= 0:
            maximum = 1.0
        fraction = min(1.0, max(0.0, float(value) / maximum))
        filled = int(round(fraction * width))
        return "█" * filled + "·" * (width - filled)

    @classmethod
    def _attention_matrix(cls, attention: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if attention is None:
            return None
        matrix = np.asarray(attention, dtype=float)
        if matrix.ndim == 4:
            matrix = matrix.mean(axis=1)
        if matrix.ndim == 3:
            matrix = matrix.mean(axis=0)
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            return None
        return matrix

    @classmethod
    def _attention_ascii(cls, attention: Optional[np.ndarray], rows: int = 8, cols: int = 32) -> list[str]:
        matrix = cls._attention_matrix(attention)
        if matrix is None:
            return ["  attention matrix unavailable"]
        if matrix.size == 0:
            return ["  attention matrix empty"]

        row_count, col_count = matrix.shape
        row_idx = np.linspace(0, row_count - 1, min(rows, row_count)).astype(int)
        col_idx = np.linspace(0, col_count - 1, min(cols, col_count)).astype(int)
        sample = matrix[np.ix_(row_idx, col_idx)]
        lo, hi = float(sample.min()), float(sample.max())
        if hi <= lo:
            normalized = np.zeros_like(sample)
        else:
            normalized = (sample - lo) / (hi - lo)
        lines = []
        for row in normalized:
            chars = "".join(
                cls._BLOCKS[min(len(cls._BLOCKS) - 1, int(v * (len(cls._BLOCKS) - 1)))]
                for v in row
            )
            lines.append("  " + chars)
        return lines

    @classmethod
    def _top_attention(cls, attention: Optional[np.ndarray], row: int, top_k: int = 6) -> str:
        matrix = cls._attention_matrix(attention)
        if matrix is None or row < 0 or row >= matrix.shape[0]:
            return "n/a"
        values = matrix[row]
        order = np.argsort(values)[::-1][:top_k]
        return ", ".join(f"{int(i) + 1}:{float(values[i]):.3f}" for i in order)

    def _render(self, telemetry: TrainingTelemetry) -> None:
        elapsed = max(1e-6, time.perf_counter() - self._started)
        width = 88
        lines = [
            "",
            "=" * width,
            "PSC CODON OPTIMIZER • CLI TRAINING OBSERVATORY",
            "=" * width,
            (
                f"epoch {telemetry.epoch}/{telemetry.total_epochs}  "
                f"step {telemetry.global_step}/{telemetry.total_steps}  "
                f"lr {telemetry.learning_rate:.3e}  "
                f"batch {telemetry.batch_time:.2f}s  "
                f"throughput {telemetry.samples_per_second:.2f} samples/s"
            ),
            "",
            "OBJECTIVES",
            f"  loss   {telemetry.loss:9.5f}  {self._sparkline(self.loss_history)}",
            f"  CE     {telemetry.ce:9.5f}  {self._sparkline(self.ce_history)}",
            f"  CAI    {telemetry.cai:9.5f}  {self._bar(telemetry.cai)}",
            f"  GC     {telemetry.gc:9.5f}  {self._bar(telemetry.gc)}",
            f"  motif  {telemetry.motif:9.5f}  {self._bar(telemetry.motif)}",
            f"  UPA    {telemetry.upa:9.5f}  {self._bar(telemetry.upa)}",
            f"  expr   {telemetry.expression_loss:9.5f}  {self._sparkline(self.expr_history)}",
            f"  grad   {telemetry.gradient_norm:9.5f}  {self._sparkline(self.grad_history)}",
            "",
            "BIOLOGICAL / EXPRESSION",
            f"  predicted expression : {telemetry.predicted_expression: .5f}",
            f"  target expression    : {telemetry.target_expression: .5f}",
            f"  expression error     : {abs(telemetry.predicted_expression - telemetry.target_expression): .5f}",
            f"  expression loss      : {telemetry.expression_loss: .5f}",
            "",
            "CODON DECODING",
            (
                f"  protein length {telemetry.protein_length} aa  |  "
                f"codons {telemetry.codon_length}  |  DNA length {3 * telemetry.codon_length} nt  |  "
                f"selected residue {telemetry.selected_position + 1}: "
                f"{telemetry.selected_amino_acid}"
            ),
        ]

        codons = list(telemetry.candidate_codons)
        probs = np.asarray(telemetry.candidate_probabilities, dtype=float)
        if codons and len(probs) == len(codons):
            order = np.argsort(probs)[::-1]
            lines.append(
                "  synonymous candidates: "
                + "  ".join(f"{codons[i]}={probs[i]:.3f}" for i in order)
            )
        else:
            lines.append("  synonymous candidates: unavailable")

        predicted = list(telemetry.predicted_codons)
        if predicted:
            preview = " ".join(predicted[:24])
            suffix = " …" if len(predicted) > 24 else ""
            lines.append(f"  predicted DNA codons : {preview}{suffix}")
        else:
            lines.append("  predicted DNA codons : unavailable")

        attention = self._attention_matrix(telemetry.cross_attention)
        lines.extend(
            [
                "",
                "DECODER → PROTEIN ATTENTION",
            ]
        )
        if attention is None:
            lines.append("  capture disabled, unavailable, or outside attention length limit")
        else:
            lines.append(f"  matrix shape          : {attention.shape[0]} × {attention.shape[1]}")
            lines.append(
                f"  selected codon top protein positions: "
                f"{self._top_attention(attention, telemetry.selected_position)}"
            )
            lines.append("  downsampled heatmap (rows=codon, cols=protein):")
            lines.extend(self._attention_ascii(telemetry.cross_attention))

        lines.extend(
            [
                "",
                "MODEL CONFIG",
                *[f"  {part}" for part in telemetry.model_config.split(", ")],
                "",
                "MODEL PARAMETERS",
                f"  total                : {telemetry.total_parameters:,}",
                f"  trainable            : {telemetry.trainable_parameters:,}",
                f"  frozen               : {max(0, telemetry.total_parameters - telemetry.trainable_parameters):,}",
                "",
                "RUNTIME",
                f"  CPU                  : {telemetry.cpu_percent:.1f}%" if telemetry.cpu_percent is not None else "  CPU                  : n/a",
                f"  RAM used             : {telemetry.memory_gb:.2f} GB" if telemetry.memory_gb is not None else "  RAM used             : n/a",
                f"  uptime               : {elapsed:.1f}s",
                f"  architecture         : {telemetry.architecture}",
                "=" * width,
            ]
        )

        # Redraw the same screen instead of appending a fresh snapshot.
        # Emit ANSI clear/home even when stdout reports non-TTY (some IDE
        # terminals and launchers misreport this); use an ANSI-capable terminal
        # for the live dashboard. CI logs may show escape sequences, but remain
        # deterministic and the ordinary non-visual training path is unchanged.
        self.stream.write("\x1b[2J\x1b[H")
        self.stream.write("\n".join(lines) + "\n")
        self.stream.flush()

    def update(self, telemetry: TrainingTelemetry):
        if self._closed:
            return
        step = telemetry.global_step
        self.step_history.append(step)
        self.loss_history.append(telemetry.loss)
        self.ce_history.append(telemetry.ce)
        self.cai_history.append(telemetry.cai)
        self.gc_history.append(telemetry.gc)
        self.motif_history.append(telemetry.motif)
        self.upa_history.append(telemetry.upa)
        self.expr_history.append(telemetry.expression_loss)
        self.grad_history.append(telemetry.gradient_norm)

        if step - self._last_render < self.update_every:
            return
        self._last_render = step
        self._render(telemetry)

    def close(self):
        self._closed = True


def candidate_codons_for_amino_acid(amino_acid: str) -> list[str]:
    return list(CODON_TABLE[amino_acid])
