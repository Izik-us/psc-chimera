"""Distribution-free calibration of fold-confidence signals against realised accuracy.

* Isotonic regression (pool-adjacent-violators) maps a raw confidence (pTM or
  mean pLDDT/100) to expected realised TM-score.
* Split-conformal, one-sided: with ``n`` exchangeable calibration residuals
  ``r_i = pred_i - true_i``, the bound ``pred - q`` with ``q`` the
  ``ceil((n+1)(1-alpha))/n`` empirical quantile satisfies
  ``P(true >= pred - q) >= 1 - alpha`` (Vovk; Lei et al.).  Mondrian groups
  (length bins) give per-group guarantees.  If a group has too few points the
  bound is ``-inf`` (fail closed), never a guess.

Exchangeability only holds if calibration data are cluster-held-out from
training *and* resemble deployment inputs; ``CalibrationState.domain`` records
what was used so the validator can refuse out-of-domain claims.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np


def expected_calibration_error(conf, observed, n_bins: int = 10) -> float:
    conf, observed = np.asarray(conf, float), np.asarray(observed, float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf >= lo) & ((conf < hi) | (hi == 1.0) & (conf <= hi))
        if sel.any():
            ece += sel.sum() / n * abs(conf[sel].mean() - observed[sel].mean())
    return float(ece)


class IsotonicCalibrator:
    """Monotone non-decreasing fit by pool-adjacent-violators, linear interpolation."""

    def __init__(self):
        self.x = self.y = None

    def fit(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        order = np.argsort(x, kind="stable")
        x, y = x[order], y[order]
        vals, wts, xs = [], [], []
        for xi, yi in zip(x, y):
            vals.append(yi), wts.append(1.0), xs.append([xi])
            while len(vals) > 1 and vals[-2] > vals[-1]:
                w = wts[-2] + wts[-1]
                vals[-2:] = [(vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w]
                wts[-2:] = [w]
                xs[-2:] = [xs[-2] + xs[-1]]
        self.x = np.array([np.mean(b) for b in xs])
        self.y = np.array(vals)
        return self

    def predict(self, x):
        if self.x is None:
            raise RuntimeError("calibrator is not fitted")
        return np.interp(np.asarray(x, float), self.x, self.y)


@dataclass
class CalibrationState:
    status: str = "uncalibrated"  # "uncalibrated" | "calibrated"
    alpha: float = 0.1
    n_calibration: int = 0
    length_edges: list = field(default_factory=lambda: [0, 200, 400, 800, 10**9])
    iso_x: list = field(default_factory=list)
    iso_y: list = field(default_factory=list)
    group_q: dict = field(default_factory=dict)  # group index -> conformal quantile (or None)
    ece_before: float | None = None
    ece_after: float | None = None
    domain: str = ""
    manifest_sha256: str = ""
    min_group: int = 20

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, default=str).encode()).hexdigest()


class ConfidenceCalibrator:
    """Predict realised TM-score from a raw confidence and give a conformal lower bound."""

    def __init__(self, state: CalibrationState | None = None):
        self.state = state or CalibrationState()
        self._iso = None
        if self.state.status == "calibrated":
            self._iso = IsotonicCalibrator()
            self._iso.x, self._iso.y = np.array(self.state.iso_x), np.array(self.state.iso_y)

    @property
    def calibrated(self) -> bool:
        return self.state.status == "calibrated"

    def _group(self, length: int) -> int:
        e = self.state.length_edges
        return max(i for i in range(len(e) - 1) if length >= e[i])

    def fit(self, raw_conf, true_tm, lengths, *, alpha=0.1, domain="", manifest_sha256="", min_group=20):
        """Fit on a *cluster-held-out* calibration set (never training data)."""
        raw, tm, L = np.asarray(raw_conf, float), np.asarray(true_tm, float), np.asarray(lengths, int)
        if len(raw) < min_group:
            raise ValueError(f"need at least {min_group} calibration examples, got {len(raw)}")
        # Split once: first half fits isotonic, second half supplies conformal residuals (keeps validity).
        idx = np.random.default_rng(0).permutation(len(raw))
        a, b = idx[: len(idx) // 2], idx[len(idx) // 2 :]
        iso = IsotonicCalibrator().fit(raw[a], tm[a])
        pred_b = iso.predict(raw[b])
        res = pred_b - tm[b]
        groups = np.array([self._group(int(l)) for l in L[b]])
        q = {}
        for g in range(len(self.state.length_edges) - 1):
            r = np.sort(res[groups == g])
            n = len(r)
            k = math.ceil((n + 1) * (1 - alpha))
            q[g] = float(r[k - 1]) if n >= min_group and k <= n else None
        self.state = CalibrationState(
            status="calibrated", alpha=alpha, n_calibration=int(len(raw)), length_edges=self.state.length_edges,
            iso_x=iso.x.tolist(), iso_y=iso.y.tolist(), group_q=q,
            ece_before=expected_calibration_error(np.clip(raw[b], 0, 1), tm[b]),
            ece_after=expected_calibration_error(np.clip(pred_b, 0, 1), tm[b]),
            domain=domain, manifest_sha256=manifest_sha256, min_group=min_group,
        )
        self._iso = iso
        return self

    def predict(self, raw_conf, length: int) -> tuple[float, float]:
        """``(expected_tm, conformal_lower_tm)``; lower is ``-inf`` if unsupported or uncalibrated."""
        if not self.calibrated:
            return float("nan"), float("-inf")
        mu = float(np.clip(self._iso.predict([raw_conf])[0], 0.0, 1.0))
        q = self.state.group_q.get(self._group(length)) if isinstance(self.state.group_q, dict) else None
        q = self.state.group_q.get(str(self._group(length))) if q is None and self.state.group_q else q
        return mu, (float("-inf") if q is None else mu - max(q, 0.0))

    def to_json(self) -> str:
        return json.dumps(asdict(self.state), sort_keys=True, default=str)

    @classmethod
    def from_json(cls, text: str) -> "ConfidenceCalibrator":
        d = json.loads(text)
        d["group_q"] = {int(k): v for k, v in d.get("group_q", {}).items()}
        return cls(CalibrationState(**d))
