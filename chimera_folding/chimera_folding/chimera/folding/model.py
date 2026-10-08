"""CHIMERAFold: sequence -> SE(3) backbone with distributional confidence.

Architecture (AlphaFold-2-style, built from CHIMERA-owned components):

    tokens -> single/pair embedding (+ relative position, + optional evo features)
           -> Pairformer trunk (triangle multiplication/attention reused from
              ``chimera.evoformer_stack``; pair-biased single attention)
           -> recycling (previous Calpha distogram bins fed back into pair)
           -> SE(3) structure module (shared-weight ``IPABlock``; frames updated on
              the group by ``R <- R exp(omega)``, ``t <- t + R dt``)
           -> confidence heads (pLDDT, PAE/pTM, distogram, carbonyl angle)

This is an *independent* sequence-only folder. It must not be given
design-derived tensors when used as a validator; ``evo_single``/``evo_pair``
exist for the training-time experiment of sharing CHIMERA's EvoFormer trunk
and are refused by ``CHIMERAFoldBackend`` for validation.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..errors import ConfigurationError
from ..evoformer_stack import (
    PairTransition,
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from ..se3_flow import IPABlock
from .frames import frames_to_backbone, so3_exp_safe
from .heads import ConfidenceHeads
from .types import PAD_IDX, FoldingBackend, FoldPrediction


@dataclass(frozen=True)
class FoldConfig:
    d_single: int = 128
    d_pair: int = 64
    n_pairformer: int = 4
    n_structure_layers: int = 8
    n_heads: int = 4
    n_tri_heads: int = 4
    c_tri_hidden: int = 32
    max_rel: int = 32
    n_recycle: int = 3
    dropout: float = 0.1
    translation_scale: float = 10.0
    recycle_bins: int = 15
    recycle_min: float = 3.375
    recycle_max: float = 21.375


class PairBiasedAttention(nn.Module):
    def __init__(self, d_single: int, d_pair: int, n_heads: int):
        super().__init__()
        if d_single % n_heads:
            raise ValueError("d_single must be divisible by n_heads")
        self.h, self.dh = n_heads, d_single // n_heads
        self.norm = nn.LayerNorm(d_single)
        self.pair_norm = nn.LayerNorm(d_pair)
        self.qkv = nn.Linear(d_single, 3 * d_single, bias=False)
        self.bias = nn.Linear(d_pair, n_heads, bias=False)
        self.gate = nn.Linear(d_single, d_single)
        self.out = nn.Linear(d_single, d_single)

    def forward(self, s, z, mask):
        B, L, _ = s.shape
        q, k, v = self.qkv(self.norm(s)).view(B, L, 3, self.h, self.dh).unbind(2)
        logits = torch.einsum("blhd,bmhd->bhlm", q, k) / self.dh**0.5
        logits = logits + self.bias(self.pair_norm(z)).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~mask[:, None, None, :], float("-inf"))
        a = logits.softmax(-1).nan_to_num(0.0)
        o = torch.einsum("bhlm,bmhd->blhd", a, v).reshape(B, L, -1)
        return self.out(torch.sigmoid(self.gate(s)) * o)


class PairformerBlock(nn.Module):
    def __init__(self, cfg: FoldConfig, depth: int):
        super().__init__()
        c = cfg
        self.attn = PairBiasedAttention(c.d_single, c.d_pair, c.n_heads)
        self.s_norm = nn.LayerNorm(c.d_single)
        self.s_ffn = nn.Sequential(nn.Linear(c.d_single, 2 * c.d_single), nn.ReLU(), nn.Linear(2 * c.d_single, c.d_single))
        self.tri_out = TriangleMultiplicationOutgoing(c.d_pair, c.c_tri_hidden, depth, c.dropout)
        self.tri_in = TriangleMultiplicationIncoming(c.d_pair, c.c_tri_hidden, depth, c.dropout)
        self.att_start = TriangleAttentionStartingNode(c.d_pair, c.n_tri_heads, depth, c.dropout)
        self.att_end = TriangleAttentionEndingNode(c.d_pair, c.n_tri_heads, depth, c.dropout)
        self.trans = PairTransition(c.d_pair, depth)

    def forward(self, s, z, mask):
        pm = (mask.unsqueeze(1) & mask.unsqueeze(2))
        z = z + self.tri_out(z, pm)
        z = z + self.tri_in(z, pm)
        z = z + self.att_start(z, pm)
        z = z + self.att_end(z, pm)
        z = z + self.trans(z)
        z = z * pm.unsqueeze(-1)
        s = s + self.attn(s, z, mask)
        s = s + self.s_ffn(self.s_norm(s))
        return s * mask.unsqueeze(-1), z


class StructureModule(nn.Module):
    """Shared-weight IPA iterations with group-valued frame updates."""

    def __init__(self, cfg: FoldConfig):
        super().__init__()
        self.cfg = cfg
        self.norm_s = nn.LayerNorm(cfg.d_single)
        self.norm_z = nn.LayerNorm(cfg.d_pair)
        self.block = IPABlock(cfg.d_single, cfg.d_pair, cfg.n_heads)
        self.update = nn.Linear(cfg.d_single, 6)
        nn.init.zeros_(self.update.weight)
        nn.init.zeros_(self.update.bias)

    def forward(self, s, z, mask):
        B, L, _ = s.shape
        s, z = self.norm_s(s), self.norm_z(z)
        R = torch.eye(3, device=s.device, dtype=s.dtype).expand(B, L, 3, 3).contiguous()
        t = torch.zeros(B, L, 3, device=s.device, dtype=s.dtype)
        traj = []
        for _ in range(self.cfg.n_structure_layers):
            s = self.block(s, z, R, t)
            upd = self.update(s) * mask.unsqueeze(-1)
            omega, dt = upd[..., :3], upd[..., 3:] * self.cfg.translation_scale
            R = R @ so3_exp_safe(omega)
            t = t + torch.einsum("blij,blj->bli", R, dt)
            traj.append((R, t))
            R = R.detach()  # stop-gradient through rotations between layers (as in AF2)
        R_final, t_final = traj[-1]
        return s, R_final, t_final, traj


class CHIMERAFold(nn.Module):
    def __init__(self, cfg: FoldConfig | None = None, d_evo_single: int | None = None, d_evo_pair: int | None = None):
        super().__init__()
        self.cfg = cfg or FoldConfig()
        c = self.cfg
        self.embed = nn.Embedding(PAD_IDX + 1, c.d_single, padding_idx=PAD_IDX)
        self.left, self.right = nn.Linear(c.d_single, c.d_pair), nn.Linear(c.d_single, c.d_pair)
        self.rel = nn.Embedding(2 * c.max_rel + 1, c.d_pair)
        self.recycle_s = nn.LayerNorm(c.d_single)
        self.recycle_z = nn.Linear(c.recycle_bins + 1, c.d_pair)
        self.evo_s = nn.Linear(d_evo_single, c.d_single) if d_evo_single else None
        self.evo_z = nn.Linear(d_evo_pair, c.d_pair) if d_evo_pair else None
        self.trunk = nn.ModuleList([PairformerBlock(c, depth=c.n_pairformer) for _ in range(c.n_pairformer)])
        self.structure = StructureModule(c)
        self.heads = ConfidenceHeads(c.d_single, c.d_pair)

    def _embed(self, tokens, mask, evo_single, evo_pair):
        c = self.cfg
        s = self.embed(tokens)
        z = self.left(s).unsqueeze(2) + self.right(s).unsqueeze(1)
        L = tokens.shape[1]
        idx = torch.arange(L, device=tokens.device)
        rel = (idx[None, :] - idx[:, None]).clamp(-c.max_rel, c.max_rel) + c.max_rel
        z = z + self.rel(rel).unsqueeze(0)
        if evo_single is not None:
            if self.evo_s is None:
                raise ConfigurationError("model was built without evo feature projections")
            s = s + self.evo_s(evo_single)
        if evo_pair is not None:
            if self.evo_z is None:
                raise ConfigurationError("model was built without evo feature projections")
            z = z + self.evo_z(evo_pair)
        return s, z

    def _recycle_pair(self, ca):
        c = self.cfg
        edges = torch.linspace(c.recycle_min, c.recycle_max, c.recycle_bins, device=ca.device, dtype=ca.dtype)
        d = torch.cdist(ca, ca)
        idx = torch.bucketize(d, edges)
        return self.recycle_z(F.one_hot(idx, c.recycle_bins + 1).to(ca.dtype))

    def forward(self, tokens, mask=None, *, n_recycle=None, evo_single=None, evo_pair=None):
        if mask is None:
            mask = tokens != PAD_IDX
        n_cycles = (self.cfg.n_recycle if n_recycle is None else n_recycle) + 1
        s0, z0 = self._embed(tokens, mask, evo_single, evo_pair)
        prev_s = torch.zeros_like(s0)
        prev_ca = torch.zeros(*tokens.shape, 3, device=tokens.device, dtype=s0.dtype)
        for cycle in range(n_cycles):
            last = cycle == n_cycles - 1
            with torch.set_grad_enabled(torch.is_grad_enabled() and last):
                s = s0 + self.recycle_s(prev_s)
                z = z0 + self._recycle_pair(prev_ca)
                for blk in self.trunk:
                    s, z = blk(s, z, mask)
                s_out, R, t, traj = self.structure(s, z, mask)
                prev_s, prev_ca = s_out.detach(), t.detach()
        h = self.heads(s_out, z)
        coords = frames_to_backbone(R, t, h["psi_sincos"])
        return {"R": R, "t": t, "trajectory": traj, "coords": coords, "single": s_out, "pair": z, **h}

    @torch.no_grad()
    def predict_tensors(self, tokens, mask=None, **kw):
        was = self.training
        self.eval()
        out = self(tokens, mask, **kw)
        self.train(was)
        mask = tokens != PAD_IDX if mask is None else mask
        out["plddt"] = self.heads.expected_plddt(out["plddt_logits"])
        out["pae"] = self.heads.expected_pae(out["pae_logits"])
        out["ptm"] = self.heads.predicted_tm(out["pae_logits"], mask)
        return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CHIMERAFoldBackend(FoldingBackend):
    """Native folder behind the sequence-only firewall; fails closed without a checkpoint."""

    name = "chimera-fold"

    def __init__(self, model: CHIMERAFold, checkpoint: str | Path | None = None, allow_untrained: bool = False):
        super().__init__()
        self.model = model
        if checkpoint is None:
            if not allow_untrained:
                raise ConfigurationError(
                    "CHIMERAFold has no trained checkpoint",
                    corrective_action="train and supply a checkpoint, or pass allow_untrained=True for plumbing tests only",
                )
            self.independence_class = "chimera_native_untrained"
            self.checkpoint_sha256 = None
        else:
            from ..backends import load_frozen_native

            load_frozen_native(model, checkpoint, "chimera-fold")
            self.independence_class = "chimera_native"
            self.checkpoint_sha256 = _sha256(Path(checkpoint))
        self.model.eval()

    def predict(self, tokens, mask=None, *, seed: int = 0, msa_tokens=None) -> FoldPrediction:
        torch.manual_seed(seed)
        if mask is None:
            mask = tokens != PAD_IDX
        # Seed diversity: recycle count jitter is deterministic given the seed.
        out = self.model.predict_tensors(tokens, mask)
        return FoldPrediction(
            tokens=tokens,
            mask=mask,
            coords=out["coords"],
            plddt=out["plddt"],
            pae=out["pae"],
            ptm=out["ptm"],
            distogram_logits=out["distogram_logits"],
            frames=(out["R"], out["t"]),
            backend=self.name,
            independence_class=self.independence_class,
            checkpoint_sha256=self.checkpoint_sha256,
            seed=seed,
            provenance={"config": asdict(self.model.cfg)},
        )
