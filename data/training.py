"""Reproducible training utilities for prepared CHIMERA JSONL data."""

import json
import hashlib
from urllib.request import urlopen
from pathlib import Path
from typing import Callable, Dict, Iterable
import torch
from torch.utils.data import DataLoader

from .dataset import ChimeraJSONLDataset


def deduplicate_records(records: Iterable[dict]) -> list[dict]:
    """Remove exact duplicate MSA records while preserving first occurrence."""
    seen = set()
    unique = []
    for record in records:
        key = json.dumps(record, sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return unique


def split_by_sequence_identity(records: list[dict], train_fraction: float = 0.8) -> tuple[list[dict], list[dict]]:
    """Deterministically split records by their first MSA sequence identity groups."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    groups: dict[str, list[dict]] = {}
    for record in records:
        sequence = tuple(record["msa_tokens"][0])
        groups.setdefault(str(sequence), []).append(record)
    ordered = list(groups.values())
    cutoff = max(1, int(len(ordered) * train_fraction))
    train = [item for group in ordered[:cutoff] for item in group]
    test = [item for group in ordered[cutoff:] for item in group]
    return train, test


def sequence_identity(left: Iterable[int], right: Iterable[int]) -> float:
    """Calculate aligned identity, excluding gap-like token 21 positions."""
    left, right = list(left), list(right)
    if len(left) != len(right):
        raise ValueError("Aligned sequences must have equal length")
    comparable = [a != 21 and b != 21 for a, b in zip(left, right)]
    return sum(a == b for a, b, keep in zip(left, right, comparable) if keep) / max(1, sum(comparable))


def cluster_split_by_identity(records: list[dict], threshold: float = 0.3, train_fraction: float = 0.8) -> tuple[list[dict], list[dict]]:
    """Greedily cluster homologous first-MSA sequences before splitting."""
    clusters: list[list[dict]] = []
    representatives: list[list[int]] = []
    for record in records:
        sequence = record["msa_tokens"][0]
        for index, representative in enumerate(representatives):
            if sequence_identity(sequence, representative) >= threshold:
                clusters[index].append(record)
                break
        else:
            representatives.append(sequence)
            clusters.append([record])
    cutoff = max(1, int(len(clusters) * train_fraction))
    return ([item for group in clusters[:cutoff] for item in group],
            [item for group in clusters[cutoff:] for item in group])


def download_manifest(url: str, destination: str | Path, sha256: str) -> Path:
    """Download a JSONL manifest and verify its SHA-256 digest."""
    destination = Path(destination)
    data = urlopen(url, timeout=60).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest.lower() != sha256.lower():
        raise ValueError(f"Checksum mismatch for {url}: expected {sha256}, got {digest}")
    destination.write_bytes(data)
    return destination


def train_epoch(model, loader: DataLoader, optimizer, loss_fn: Callable) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model, batch)
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
    return total / max(1, len(loader))


def save_checkpoint(path: str | Path, model, optimizer, epoch: int, metrics: Dict) -> None:
    """Save model, optimizer, epoch, and metrics as one atomic checkpoint."""
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "metrics": metrics}, temporary)
    temporary.replace(target)


def make_loader(path: str | Path, batch_size: int = 1, shuffle: bool = True) -> DataLoader:
    return DataLoader(ChimeraJSONLDataset(path), batch_size=batch_size, shuffle=shuffle)
