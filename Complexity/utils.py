from __future__ import annotations

from typing import Optional
import math

import torch
import torch.nn.functional as F


def standardize(z: torch.Tensor) -> torch.Tensor:
    return (z - z.mean(dim=-2, keepdim=True)) / (z.std(dim=-2, keepdim=True) + 1e-8)


def split_indices(n: int, train_frac: float, seed: int):
    if n <= 1:
        return torch.arange(n), torch.empty(0, dtype=torch.long)
    g = torch.Generator()
    g.manual_seed(int(seed))
    perm = torch.randperm(n, generator=g)
    n_train = min(max(1, int(round(float(train_frac) * n))), n - 1)
    return perm[:n_train], perm[n_train:]


def r2(y: torch.Tensor, pred: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    y, pred = y.view(-1).float(), pred.view(-1).float()
    return (1.0 - (y - pred).pow(2).mean() / ((y - baseline).pow(2).mean() + 1e-8)).clamp(0, 1)


def r2_vec(y: torch.Tensor, pred: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
    mse0 = (y.float() - baseline.float()).pow(2).mean()
    if mse0 < 1e-8:
        return torch.tensor(0.0, device=y.device)
    return (1.0 - (y.float() - pred.float()).pow(2).mean() / (mse0 + 1e-8)).clamp(0, 1)


def acc_to_01(acc: torch.Tensor) -> torch.Tensor:
    return (2.0 * acc - 1.0).clamp(0.0, 1.0)


def train(model, x, y, *, loss: str, steps: int, lr: float):
    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss_value = F.mse_loss(out, y) if loss == "mse" else F.cross_entropy(out, y)
        loss_value.backward()
        opt.step()


def windows(z: torch.Tensor, k: int):
    if z.shape[0] <= k:
        raise ValueError("Need T > k")
    w = z.unfold(0, k, 1)
    return w[:-1].permute(0, 2, 1), z[k:], w[1:].permute(0, 2, 1), z[:-k]


def categorical_entropy(x: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    p = F.one_hot(x.long().clamp(0, k - 1), k).float().mean(dim=dim).clamp_min(1e-8)
    return -(p * torch.log2(p)).sum(dim=-1) / max(math.log2(k), 1.0)


def num_states(z: torch.Tensor, explicit: Optional[int], fallback=None) -> int:
    if explicit is not None:
        return int(explicit)
    if fallback is not None:
        if isinstance(fallback, int):
            return int(fallback)
        return max(int(v) for v in fallback)
    return int(z.max().item()) + 1 if z.numel() else 1


def state_bytes(k: int) -> int:
    return max(1, ((max(1, int(k) - 1).bit_length()) + 7) // 8)


def encode_states(x: torch.Tensor, k: int) -> bytes:
    x = x.reshape(-1).detach().cpu().long()
    if x.numel() == 0:
        return b""
    bps = state_bytes(k)
    if bps == 1:
        return x.to(torch.uint8).contiguous().numpy().tobytes()
    return b"".join(int(v).to_bytes(bps, "little", signed=False) for v in x.tolist())


def brotli_compress(payload: bytes, quality: Optional[int]) -> bytes:
    import brotli

    return brotli.compress(payload) if quality is None else brotli.compress(payload, quality=int(quality))
