from __future__ import annotations

import torch

from .nn import MLP, TinyTransformerClassifier
from .base import ComplexityMetric
from .utils import acc_to_01, split_indices, standardize, train


class PairOrderMLPClassifier(ComplexityMetric):
    def __init__(self, train_frac: float = 0.5, steps: int = 200, lr: float = 1e-2, hidden: int = 128, depth: int = 2, act: str = "relu", standardize: bool = True, seed: int = 0):
        self.train_frac, self.steps, self.lr, self.hidden, self.depth, self.act, self.standardize, self.seed = float(train_frac), int(steps), float(lr), int(hidden), int(depth), act, bool(standardize), int(seed)

    def score_one(self, embedding, i: int) -> torch.Tensor:
        x = embedding.tensor[i].detach().float()
        return _pair_order(x, self.seed + 1000 * int(i), self, mlp=True)


class PairOrderKNNClassifier(ComplexityMetric):
    def __init__(self, train_frac: float = 0.5, standardize: bool = True, seed: int = 0):
        self.train_frac, self.standardize, self.seed = float(train_frac), bool(standardize), int(seed)

    def score_one(self, embedding, i: int) -> torch.Tensor:
        x = embedding.tensor[i].detach().float()
        return _pair_order(x, self.seed + 1000 * int(i), self, mlp=False)


class ContextFuturePastMLPClassifier(ComplexityMetric):
    def __init__(self, k: int = 3, train_frac: float = 0.5, steps: int = 200, lr: float = 1e-2, hidden: int = 128, depth: int = 2, act: str = "relu", standardize: bool = True, seed: int = 0):
        self.k, self.train_frac, self.steps, self.lr, self.hidden, self.depth, self.act, self.standardize, self.seed = int(k), float(train_frac), int(steps), float(lr), int(hidden), int(depth), act, bool(standardize), int(seed)

    def score_one(self, embedding, i: int) -> torch.Tensor:
        x = embedding.tensor[i].detach().float()
        seed = self.seed + 1000 * int(i)
        T = x.shape[0]
        if T < self.k + 3:
            return torch.tensor(0.0, device=x.device)
        z = standardize(x) if self.standardize else x
        X, y = [], []
        for t in range(1, T - self.k):
            ctx = z[t : t + self.k].reshape(-1)
            X += [torch.cat([ctx, z[t - 1]]), torch.cat([ctx, z[t + self.k]])]
            y += [0, 1]
        return _mlp_classifier(torch.stack(X).float(), torch.tensor(y, device=x.device), seed, self)


class OrderedVsShuffledTransformer(ComplexityMetric):
    def __init__(self, L: int = 8, train_frac: float = 0.5, steps: int = 300, lr: float = 1e-3, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, standardize: bool = True, seed: int = 0):
        self.L, self.train_frac, self.steps, self.lr, self.d_model, self.n_heads, self.n_layers, self.standardize, self.seed = int(L), float(train_frac), int(steps), float(lr), int(d_model), int(n_heads), int(n_layers), bool(standardize), int(seed)

    def score_one(self, embedding, i: int) -> torch.Tensor:
        x = embedding.tensor[i].detach().float()
        seed = self.seed + 1000 * int(i)
        T, D = x.shape
        if T < self.L + 2:
            return torch.tensor(0.0, device=x.device)
        z = standardize(x) if self.standardize else x
        g = torch.Generator().manual_seed(seed)
        X, y = [], []
        for t in range(T - self.L + 1):
            w = z[t : t + self.L]
            X += [w, w[torch.randperm(self.L, generator=g)]]
            y += [1, 0]
        X, y = torch.stack(X).float(), torch.tensor(y, device=x.device)
        tr, te = split_indices(X.shape[0], self.train_frac, seed)
        if te.numel() == 0:
            return torch.tensor(0.0, device=x.device)
        tr, te = tr.to(x.device), te.to(x.device)
        model = TinyTransformerClassifier(D, 2, self.d_model, self.n_heads, self.n_layers).to(x.device)
        train(model, X[tr], y[tr], loss="ce", steps=self.steps, lr=self.lr)
        with torch.no_grad():
            return acc_to_01((model(X[te]).argmax(dim=1) == y[te]).float().mean())


def _pair_order(x: torch.Tensor, seed: int, cfg, *, mlp: bool) -> torch.Tensor:
    T = x.shape[0]
    if T < 4:
        return torch.tensor(0.0, device=x.device)
    z = standardize(x) if cfg.standardize else x
    a, b = z[:-1], z[1:]
    X = torch.cat([torch.cat([a, b], dim=1), torch.cat([b, a], dim=1)]).float()
    y = torch.cat([torch.ones(T - 1, device=x.device, dtype=torch.long), torch.zeros(T - 1, device=x.device, dtype=torch.long)])
    if mlp:
        return _mlp_classifier(X, y, seed, cfg)
    tr, te = split_indices(X.shape[0], cfg.train_frac, seed)
    if te.numel() == 0:
        return torch.tensor(0.0, device=x.device)
    tr, te = tr.to(x.device), te.to(x.device)
    pred = y[tr][torch.cdist(X[te], X[tr]).argmin(dim=1)]
    return acc_to_01((pred == y[te]).float().mean())


def _mlp_classifier(X: torch.Tensor, y: torch.Tensor, seed: int, cfg) -> torch.Tensor:
    tr, te = split_indices(X.shape[0], cfg.train_frac, seed)
    if te.numel() == 0:
        return torch.tensor(0.0, device=X.device)
    tr, te = tr.to(X.device), te.to(X.device)
    model = MLP(X.shape[-1], 2, cfg.hidden, cfg.depth, cfg.act).to(X.device)
    train(model, X[tr], y[tr], loss="ce", steps=cfg.steps, lr=cfg.lr)
    with torch.no_grad():
        return acc_to_01((model(X[te]).argmax(dim=1) == y[te]).float().mean())
