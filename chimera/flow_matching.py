"""Compatibility exports for historical flow-matching import paths.

Canonical structural generation is implemented in :mod:`chimera.se3_flow`;
legacy deterministic OT utilities are isolated in :mod:`chimera.legacy_flow`.
"""

from .lie import so3_exp, so3_log
from .legacy_flow import SE3FlowMatching, se3_interp, so3_geodesic_interp, so3_velocity
from .se3_flow import (
    FlowMatchingBackbone,
    IPABlock,
    InvariantPointAttention,
    SinusoidalTimeEmbedding,
    VelocityField,
)

__all__ = [
    "FlowMatchingBackbone",
    "IPABlock",
    "InvariantPointAttention",
    "SE3FlowMatching",
    "SinusoidalTimeEmbedding",
    "VelocityField",
    "se3_interp",
    "so3_exp",
    "so3_geodesic_interp",
    "so3_log",
    "so3_velocity",
]
