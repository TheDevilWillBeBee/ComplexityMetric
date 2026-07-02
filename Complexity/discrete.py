from __future__ import annotations

from typing import Optional
import math

import torch
import torch.nn.functional as F

from .base import ComplexityMetric
from .utils import brotli_compress, categorical_entropy, encode_states, num_states as infer_num_states, state_bytes


class Entropy(ComplexityMetric):
    input_type = "discrete"

    def __init__(self, probability_mode: str = "time", num_states: Optional[int] = None):
        self.probability_mode = _mode(probability_mode)
        self.num_states = num_states

    def score_one(self, embedding, i: int) -> torch.Tensor:
        x = embedding.tensor[i].detach().long()
        k = infer_num_states(x, self.num_states, embedding.num_states)
        if k <= 1:
            return torch.tensor(0.0, device=x.device)
        if self.probability_mode == "time":
            return categorical_entropy(x, k, dim=0).mean().clamp(0, 1)
        if self.probability_mode == "space":
            return categorical_entropy(x, k, dim=1).mean().clamp(0, 1)
        return categorical_entropy(x.reshape(-1), k, dim=0).clamp(0, 1)


class FutureStateMutualInformation(ComplexityMetric):
    input_type = "discrete"

    def __init__(self, probability_mode: str = "time", num_states: Optional[int] = None):
        self.probability_mode = _mode(probability_mode)
        self.num_states = num_states

    def score_one(self, embedding, i: int) -> torch.Tensor:
        x = embedding.tensor[i].detach().long()
        T, D = x.shape
        if T < 2:
            return torch.tensor(0.0, device=x.device)
        k = infer_num_states(x, self.num_states, embedding.num_states)
        if k <= 1:
            return torch.tensor(0.0, device=x.device)

        a = F.one_hot(x[:-1].clamp(0, k - 1), k).float()
        b = F.one_hot(x[1:].clamp(0, k - 1), k).float()
        if self.probability_mode == "time":
            return _mi(torch.einsum("tdi,tdj->dij", a, b) / (T - 1), k).mean().clamp(0, 1)
        if self.probability_mode == "space":
            return _mi(torch.einsum("tdi,tdj->tij", a, b) / max(D, 1), k).mean().clamp(0, 1)
        a, b = a.reshape(-1, k), b.reshape(-1, k)
        return _mi(torch.einsum("ni,nj->ij", a, b) / max(a.shape[0], 1), k).clamp(0, 1)


class CompressionComplexity(ComplexityMetric):
    input_type = "discrete"
    ROW_MAJOR = "row_major"
    COLUMN_MAJOR = "column_major"
    MIN_COMPRESSION = "min_compression"
    default_flatten_order = MIN_COMPRESSION

    def __init__(self, num_states: Optional[int] = None, flatten_order: Optional[str] = None, quality: Optional[int] = None):
        self.num_states = num_states
        self.flatten_order = _order(self.default_flatten_order if flatten_order is None else flatten_order)
        self.quality = quality

    def score_one(self, embedding, i: int) -> torch.Tensor:
        x = embedding.tensor[i].detach().long()
        k = infer_num_states(x, self.num_states, embedding.num_states)
        if k <= 1 or x.numel() == 0:
            return torch.tensor(0.0, device=x.device)

        values = [_compressed_ratio(x, k, self.quality, order) for order in _flatten_orders(self.flatten_order)]
        value = min(values) if values else 0.0
        return torch.tensor(value, device=x.device, dtype=torch.float32)


def _compressed_ratio(x: torch.Tensor, k: int, quality: Optional[int], flatten_order: str) -> float:
    flat = x.reshape(-1) if flatten_order == CompressionComplexity.ROW_MAJOR else x.transpose(0, 1).reshape(-1)
    payload = encode_states(flat, k)
    expected = math.log2(k) / (8.0 * state_bytes(k))
    return 0.0 if not payload or expected <= 0 else len(brotli_compress(payload, quality)) / len(payload) / expected


class CompressedRatio(CompressionComplexity):
    pass


class EntropyMinusCompressedRatio(ComplexityMetric):
    input_type = "discrete"

    def __init__(self):
        self.compressed_ratio = CompressedRatio()
        self.entropy = Entropy()

    def score(self, embedding) -> torch.Tensor:
        return self.entropy.score(embedding) - self.compressed_ratio.score(embedding)


CompressedRatioMinusEntropy = EntropyMinusCompressedRatio


class DensityTransientTime(ComplexityMetric):
    """First time the mean state value reaches and stays near its tail mean."""

    input_type = "discrete"

    def __init__(
        self,
        tail_length: int = 256,
        epsilon: float = 5e-2,
        confirmation_window: int = 100,
        tail_std_scale: float = 1.0,
        normalize_time: bool = True,
        debug: bool = False,
    ):
        self.tail_length = int(tail_length)
        if self.tail_length < 1:
            raise ValueError("tail_length must be >= 1")

        self.epsilon = float(epsilon)
        if self.epsilon < 0:
            raise ValueError("epsilon must be >= 0")

        self.confirmation_window = int(confirmation_window)
        if self.confirmation_window < 1:
            raise ValueError("confirmation_window must be >= 1")

        self.tail_std_scale = float(tail_std_scale)
        if self.tail_std_scale < 0:
            raise ValueError("tail_std_scale must be >= 0")

        self.normalize_time = bool(normalize_time)
        self.debug = bool(debug)

    def score(self, embedding) -> torch.Tensor:
        x = embedding.tensor.detach().float() # [B, T, D]
        if x.shape[-1] == 0:
            return torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)

        density = x.mean(dim=-1)
        _, T = density.shape
        if T == 0:
            return torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)

        tail = density[:, -min(self.tail_length, T) :]
        tail_mean = tail.mean(dim=1)
        tail_std = tail.std(dim=1, unbiased=False)
        tolerance = tail_std * self.tail_std_scale + self.epsilon

        lifetime = _first_entry_time_to_tail_mean(
            density,
            tail_mean,
            tolerance,
            self.confirmation_window,
        )
        if self.normalize_time:
            lifetime = lifetime / float(max(T, 1))
        return lifetime.to(dtype=torch.float32)


def _mi(joint: torch.Tensor, k: int) -> torch.Tensor:
    joint = joint.clamp_min(1e-8)
    px = joint.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    py = joint.sum(dim=-2, keepdim=True).clamp_min(1e-8)
    return (joint * torch.log2(joint / (px * py))).sum(dim=(-1, -2)) / max(math.log2(k), 1.0)


def _first_entry_time_to_tail_mean(
    series: torch.Tensor,
    target: torch.Tensor,
    tolerance: torch.Tensor,
    confirmation_window: int,
) -> torch.Tensor:
    if series.ndim != 2:
        raise ValueError("series must have shape (B,T)")
    if target.ndim != 1 or target.shape[0] != series.shape[0]:
        raise ValueError("target must have shape (B,)")
    if tolerance.ndim != 1 or tolerance.shape[0] != series.shape[0]:
        raise ValueError("tolerance must have shape (B,)")

    B, T = series.shape
    W = int(confirmation_window)
    if T < W:
        return torch.full((B,), float(T), device=series.device)

    dev = (series - target.unsqueeze(1)).abs()
    out = torch.full((B,), float(T), device=series.device)
    found = torch.zeros(B, dtype=torch.bool, device=series.device)

    for t in range(T - W + 1):
        window_ok = (dev[:, t : t + W] <= tolerance.unsqueeze(1)).all(dim=1)
        newly_found = window_ok & ~found
        if newly_found.any():
            out[newly_found] = float(t)
            found[newly_found] = True
        if found.all():
            break

    return out


def _mode(mode: str) -> str:
    mode = str(mode).lower()
    if mode not in ("time", "space", "both"):
        raise ValueError("probability_mode must be 'time', 'space', or 'both'")
    return mode


def _order(order: str) -> str:
    order = str(order).lower().replace("-", "_")
    if order in ("row", "rowmajor", "row_major"):
        return CompressionComplexity.ROW_MAJOR
    if order in ("col", "column", "columnmajor", "col_major", "column_major"):
        return CompressionComplexity.COLUMN_MAJOR
    if order in ("auto", "both", "best", "min", "minimum", "min_compression", "minimum_compression"):
        return CompressionComplexity.MIN_COMPRESSION
    raise ValueError("flatten_order must be row_major, column_major, or min_compression")


def _flatten_orders(flatten_order: str) -> tuple[str, ...]:
    if flatten_order == CompressionComplexity.MIN_COMPRESSION:
        return (CompressionComplexity.ROW_MAJOR, CompressionComplexity.COLUMN_MAJOR)
    return (flatten_order,)
