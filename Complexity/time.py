from __future__ import annotations

import torch
import torch.nn.functional as F

from .nn import MLP
from .base import ComplexityMetric
from .utils import r2, split_indices, standardize, train


class OpenEndedness(ComplexityMetric):
    def score_one(self, embedding, i: int) -> torch.Tensor:
        x = embedding.tensor[i].detach().float()
        sim = F.cosine_similarity(x[:, None], x[None], dim=-1)
        sim = torch.tril(sim, diagonal=-1)
        return (1.0 - sim.max()).clamp_min(0)


class LinearRidgeTimeRegression(ComplexityMetric):
    def __init__(self, train_frac: float = 0.5, ridge: float = 1e-3, standardize: bool = True, seed: int = 0):
        self.train_frac, self.ridge, self.standardize, self.seed = float(train_frac), float(ridge), bool(standardize), int(seed)

    def score_one(self, embedding, i: int) -> torch.Tensor:
        x = embedding.tensor[i].detach().float()
        seed = self.seed + 1000 * int(i)
        T, D = x.shape
        if T < 4:
            return torch.tensor(0.0, device=x.device)
        z = standardize(x) if self.standardize else x
        y = torch.linspace(0, 1, T, device=x.device)
        tr, te = split_indices(T, self.train_frac, seed)
        if te.numel() == 0:
            return torch.tensor(0.0, device=x.device)
        tr, te = tr.to(x.device), te.to(x.device)
        Xtr = torch.cat([z[tr], torch.ones(tr.numel(), 1, device=x.device)], dim=1)
        Xte = torch.cat([z[te], torch.ones(te.numel(), 1, device=x.device)], dim=1)
        A = Xtr.T @ Xtr + torch.eye(D + 1, device=x.device) * self.ridge
        w = torch.linalg.solve(A, Xtr.T @ y[tr, None])
        return r2(y[te], (Xte @ w).squeeze(1), y[tr].mean())


class KNNTimeRegression(ComplexityMetric):
    def __init__(self, train_frac: float = 0.5, standardize: bool = True, seed: int = 0):
        self.train_frac, self.standardize, self.seed = float(train_frac), bool(standardize), int(seed)

    def score_one(self, embedding, i: int) -> torch.Tensor:
        x = embedding.tensor[i].detach().float()
        seed = self.seed + 1000 * int(i)
        T = x.shape[0]
        if T < 6:
            return torch.tensor(0.0, device=x.device)
        z = standardize(x) if self.standardize else x
        y = torch.linspace(0, 1, T, device=x.device)
        tr, te = split_indices(T, self.train_frac, seed)
        if te.numel() == 0:
            return torch.tensor(0.0, device=x.device)
        tr, te = tr.to(x.device), te.to(x.device)
        pred = y[tr][torch.cdist(z[te].float(), z[tr].float()).argmin(dim=1)]
        return r2(y[te], pred, y[tr].mean())


class MLPTimeRegression(ComplexityMetric):
    def __init__(self, train_frac: float = 0.5, steps: int = 200, lr: float = 1e-2, hidden: int = 128, depth: int = 2, act: str = "relu", standardize: bool = True, seed: int = 0):
        self.train_frac, self.steps, self.lr, self.hidden, self.depth, self.act, self.standardize, self.seed = float(train_frac), int(steps), float(lr), int(hidden), int(depth), act, bool(standardize), int(seed)

    def score_one(self, embedding, i: int) -> torch.Tensor:
        x = embedding.tensor[i].detach().float()
        seed = self.seed + 1000 * int(i)
        T, D = x.shape
        if T < 6:
            return torch.tensor(0.0, device=x.device)
        z = standardize(x) if self.standardize else x
        y = torch.linspace(0, 1, T, device=x.device).float()
        tr, te = split_indices(T, self.train_frac, seed)
        if te.numel() == 0:
            return torch.tensor(0.0, device=x.device)
        tr, te = tr.to(x.device), te.to(x.device)
        model = MLP(D, 1, self.hidden, self.depth, self.act).to(x.device)
        train(model, z[tr].float(), y[tr, None], loss="mse", steps=self.steps, lr=self.lr)
        with torch.no_grad():
            pred = model(z[te].float()).squeeze(1)
        return r2(y[te], pred, y[tr].mean())
