"""Reference-faithful wrapper for the upstream Dauparas ProteinMPNN.

The upstream implementation is vendored under chimera.vendor with its MIT
license and source revision. This wrapper keeps the native module/state-dict
layout intact so official model_state_dict checkpoints can be loaded strictly.
CHIMERA conditioning is intentionally kept outside the reference core.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

from .vendor.proteinmpnn_utils import ProteinMPNN as _UpstreamProteinMPNN


AMINO_ACID_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
REFERENCE_HIDDEN_DIM = 128
REFERENCE_LAYERS = 3
REFERENCE_NEIGHBORS = 48


class ProteinMPNNReference(nn.Module):
    """Original ProteinMPNN architecture with a CHIMERA-friendly input contract.

    backbone_coords must use atom order N, CA, C, O and shape (B,L,4,3)
    (a fifth atom, if present, is ignored). sequence uses the upstream
    21-token alphabet ACDEFGHIKLMNPQRSTVWYX. design_mask is one for positions
    to redesign and zero for positions to keep fixed.

    The wrapped upstream model remains independently usable and is exposed as
    native_model. Its parameter names are not renamed or projected.
    """

    def __init__(
        self,
        hidden_dim: int = REFERENCE_HIDDEN_DIM,
        num_encoder_layers: int = REFERENCE_LAYERS,
        num_decoder_layers: int = REFERENCE_LAYERS,
        k_neighbors: int = REFERENCE_NEIGHBORS,
        augment_eps: float = 0.0,
        dropout: float = 0.1,
        ca_only: bool = False,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if num_encoder_layers < 1 or num_decoder_layers < 1:
            raise ValueError("encoder and decoder layer counts must be positive")
        if k_neighbors < 1:
            raise ValueError("k_neighbors must be positive")
        if augment_eps < 0:
            raise ValueError("augment_eps cannot be negative")
        self.hidden_dim = int(hidden_dim)
        self.num_encoder_layers = int(num_encoder_layers)
        self.num_decoder_layers = int(num_decoder_layers)
        self.k_neighbors = int(k_neighbors)
        self.augment_eps = float(augment_eps)
        self.ca_only = bool(ca_only)
        self.native_model = _UpstreamProteinMPNN(
            num_letters=21,
            node_features=self.hidden_dim,
            edge_features=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            vocab=21,
            k_neighbors=self.k_neighbors,
            augment_eps=self.augment_eps,
            dropout=dropout,
            ca_only=self.ca_only,
        )

    @staticmethod
    def _prepare_inputs(
        backbone_coords: torch.Tensor,
        sequence: torch.Tensor,
        residue_mask: torch.Tensor,
        design_mask: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if backbone_coords.ndim != 4 or backbone_coords.shape[-1] != 3:
            raise ValueError("backbone_coords must have shape (B, L, 4|5, 3)")
        if backbone_coords.shape[-2] not in (4, 5):
            raise ValueError("backbone_coords must contain N, CA, C, O (and optionally CB)")
        batch, length = backbone_coords.shape[:2]
        expected = (batch, length)
        for name, value in (
            ("sequence", sequence),
            ("residue_mask", residue_mask),
            ("design_mask", design_mask),
            ("residue_idx", residue_idx),
            ("chain_encoding", chain_encoding),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape (B, L)")
        if sequence.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("sequence must contain integer amino-acid token IDs")
        if torch.any((sequence < 0) | (sequence >= len(AMINO_ACID_ALPHABET))):
            raise ValueError("sequence token IDs must be in [0, 20]")
        if residue_mask.dtype != torch.bool:
            raise ValueError("residue_mask must have dtype bool")
        if design_mask.dtype != torch.bool:
            raise ValueError("design_mask must have dtype bool")
        if residue_idx.dtype not in (torch.int32, torch.int64):
            raise ValueError("residue_idx must have integer dtype")
        if chain_encoding.dtype not in (torch.int32, torch.int64):
            raise ValueError("chain_encoding must have integer dtype")
        if torch.any(design_mask & ~residue_mask):
            raise ValueError("design_mask cannot include padded or missing residues")
        coords = backbone_coords[..., :4, :]
        valid_atoms = residue_mask[..., None, None].expand_as(coords)
        if not torch.isfinite(coords[valid_atoms]).all():
            raise ValueError("valid backbone coordinates must be finite")
        # Upstream distance featurization can propagate NaNs from padding even
        # when the residue mask is zero, so padding coordinates are made finite.
        coords = torch.where(valid_atoms, coords, torch.zeros_like(coords))
        mask = residue_mask.to(dtype=coords.dtype)
        chain_mask = design_mask.to(dtype=coords.dtype)
        return (
            coords,
            sequence.to(dtype=torch.long),
            mask,
            chain_mask,
            residue_idx.to(dtype=torch.long),
            chain_encoding.to(dtype=torch.long),
        )

    @staticmethod
    def _validate_decoding_order(
        decoding_order: Optional[torch.Tensor], batch: int, length: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        if decoding_order is None:
            return None
        if decoding_order.shape != (batch, length):
            raise ValueError("decoding_order must have shape (B, L)")
        if decoding_order.dtype not in (torch.int32, torch.int64):
            raise ValueError("decoding_order must have integer dtype")
        expected = torch.arange(length, device=device).expand(batch, -1)
        actual = torch.sort(decoding_order.to(device=device, dtype=torch.long), dim=-1).values
        if not torch.equal(actual, expected):
            raise ValueError("each decoding_order row must be a permutation of 0..L-1")
        return decoding_order.to(device=device, dtype=torch.long)

    def forward(
        self,
        backbone_coords: torch.Tensor,
        sequence: torch.Tensor,
        residue_mask: torch.Tensor,
        design_mask: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding: torch.Tensor,
        randn: Optional[torch.Tensor] = None,
        decoding_order: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return native ProteinMPNN log probabilities with shape (B,L,21).

        Supply randn and optionally decoding_order to reproduce a particular
        upstream decoding context. With neither supplied, the upstream
        randomized autoregressive-order behavior is used.
        """
        X, S, mask, chain_M, residue_idx, chain_encoding = self._prepare_inputs(
            backbone_coords, sequence, residue_mask, design_mask, residue_idx, chain_encoding
        )
        batch, length = S.shape
        if self.ca_only:
            X = X[:, :, 1, :]
        if randn is None:
            randn = torch.randn((batch, length), device=X.device, dtype=X.dtype)
        elif randn.shape != (batch, length):
            raise ValueError("randn must have shape (B, L)")
        order = self._validate_decoding_order(decoding_order, batch, length, X.device)
        return self.native_model(
            X,
            S,
            mask,
            chain_M,
            residue_idx,
            chain_encoding,
            randn,
            use_input_decoding_order=order is not None,
            decoding_order=order,
        )

    @torch.no_grad()
    def sample(
        self,
        backbone_coords: torch.Tensor,
        sequence: torch.Tensor,
        residue_mask: torch.Tensor,
        design_mask: torch.Tensor,
        residue_idx: torch.Tensor,
        chain_encoding: torch.Tensor,
        temperature: float = 0.1,
        randn: Optional[torch.Tensor] = None,
        omit_aas: Optional[np.ndarray] = None,
        amino_acid_bias: Optional[np.ndarray] = None,
        omit_aa_mask: Optional[torch.Tensor] = None,
        bias_by_residue: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Sample using the upstream autoregressive sampler.

        Fixed positions retain their input tokens. Optional omit/bias arrays
        follow upstream ProteinMPNN's 21-token alphabet and conventions.
        """
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        X, S, mask, chain_M, residue_idx, chain_encoding = self._prepare_inputs(
            backbone_coords, sequence, residue_mask, design_mask, residue_idx, chain_encoding
        )
        batch, length = S.shape
        if self.ca_only:
            X = X[:, :, 1, :]
        if randn is None:
            randn = torch.randn((batch, length), device=X.device, dtype=X.dtype)
        elif randn.shape != (batch, length):
            raise ValueError("randn must have shape (B, L)")
        omit = np.zeros(21, dtype=np.float32) if omit_aas is None else np.asarray(omit_aas, dtype=np.float32)
        bias = np.zeros(21, dtype=np.float32) if amino_acid_bias is None else np.asarray(amino_acid_bias, dtype=np.float32)
        if omit.shape != (21,) or bias.shape != (21,):
            raise ValueError("omit_aas and amino_acid_bias must each have shape (21,)")
        if bias_by_residue is None:
            bias_by_residue = torch.zeros((batch, length, 21), device=X.device, dtype=X.dtype)
        elif bias_by_residue.shape != (batch, length, 21):
            raise ValueError("bias_by_residue must have shape (B, L, 21)")
        if omit_aa_mask is not None and omit_aa_mask.shape != (batch, length, 21):
            raise ValueError("omit_aa_mask must have shape (B, L, 21)")
        return self.native_model.sample(
            X=X,
            randn=randn,
            S_true=S,
            chain_mask=chain_M,
            chain_encoding_all=chain_encoding,
            residue_idx=residue_idx,
            mask=mask,
            temperature=temperature,
            omit_AAs_np=omit,
            bias_AAs_np=bias,
            chain_M_pos=torch.ones_like(chain_M),
            omit_AA_mask=omit_aa_mask,
            pssm_coef=None,
            pssm_bias=None,
            pssm_multi=0.0,
            pssm_log_odds_flag=False,
            pssm_log_odds_mask=None,
            pssm_bias_flag=False,
            bias_by_res=bias_by_residue,
        )

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        *,
        map_location: str | torch.device = "cpu",
        augment_eps: float = 0.0,
        dropout: float = 0.1,
        ca_only: bool = False,
        strict: bool = True,
    ) -> "ProteinMPNNReference":
        """Load an official checkpoint without renaming or reshaping weights."""
        path = Path(checkpoint_path).expanduser()
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        if not isinstance(checkpoint, dict):
            raise ValueError("ProteinMPNN checkpoint must be a dictionary")
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        if not isinstance(state, dict):
            raise ValueError("checkpoint does not contain a model state dictionary")
        hidden_dim = 128
        first_weight = state.get("W_e.weight")
        if first_weight is not None:
            hidden_dim = int(first_weight.shape[0])
        num_edges = int(checkpoint.get("num_edges", REFERENCE_NEIGHBORS))
        # The official checkpoint stores graph neighborhood size but not every
        # constructor argument. The published standard architecture uses 3+3 layers.
        model = cls(
            hidden_dim=hidden_dim,
            num_encoder_layers=REFERENCE_LAYERS,
            num_decoder_layers=REFERENCE_LAYERS,
            k_neighbors=num_edges,
            augment_eps=augment_eps,
            dropout=dropout,
            ca_only=ca_only,
        )
        model.native_model.load_state_dict(state, strict=strict)
        model.eval()
        return model
