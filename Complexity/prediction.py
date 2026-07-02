from __future__ import annotations

import torch

from .nn import MLP, TinyTransformerRegressor
from .base import ComplexityMetric
from .utils import r2_vec, split_indices, standardize, train, windows


class _ForwardBackwardNextStepAOTBase(ComplexityMetric):
    def __init__(self, k: int = 4, train_frac: float = 0.5, standardize: bool = True, max_samples: int | None = None, seed: int = 0):
        self.k, self.train_frac, self.standardize, self.max_samples, self.seed = int(k), float(train_frac), bool(standardize), max_samples, int(seed)

    def score_one(self, embedding, i: int) -> torch.Tensor:
        x = embedding.tensor[i].detach().float()
        seed = self.seed + 1000 * int(i)
        if x.shape[0] < self.k + 3:
            return torch.tensor(0.0, device=x.device)
        z = standardize(x) if self.standardize else x
        cf, yf, cb, yb = windows(z, self.k)
        n = cf.shape[0]
        if self.max_samples is not None and n > self.max_samples:
            idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[: self.max_samples].to(x.device)
            cf, yf, cb, yb, n = cf[idx], yf[idx], cb[idx], yb[idx], idx.numel()
        tr, te = split_indices(n, self.train_frac, seed)
        if te.numel() == 0:
            return torch.tensor(0.0, device=x.device)
        tr, te = tr.to(x.device), te.to(x.device)
        return (self.fit_r2(cf[tr], yf[tr], cf[te], yf[te]) - self.fit_r2(cb[tr], yb[tr], cb[te], yb[te])).abs().clamp(0, 1)

    def fit_r2(self, ctx_tr, y_tr, ctx_te, y_te):
        raise NotImplementedError


class ForwardBackwardMLPNextStepAOT(_ForwardBackwardNextStepAOTBase):
    def __init__(self, k: int = 4, train_frac: float = 0.5, steps: int = 200, lr: float = 1e-2, hidden: int = 128, depth: int = 2, act: str = "relu", standardize: bool = True, max_samples: int | None = None, seed: int = 0):
        super().__init__(k, train_frac, standardize, max_samples, seed)
        self.steps, self.lr, self.hidden, self.depth, self.act = int(steps), float(lr), int(hidden), int(depth), act

    def fit_r2(self, ctx_tr, y_tr, ctx_te, y_te):
        _, k, D = ctx_tr.shape
        Xtr, Xte = ctx_tr.reshape(ctx_tr.shape[0], k * D).float(), ctx_te.reshape(ctx_te.shape[0], k * D).float()
        model = MLP(k * D, D, self.hidden, self.depth, self.act).to(ctx_tr.device)
        train(model, Xtr, y_tr.float(), loss="mse", steps=self.steps, lr=self.lr)
        with torch.no_grad():
            pred = model(Xte)
        return r2_vec(y_te, pred, y_tr.mean(dim=0, keepdim=True))


class ForwardBackwardTransformerNextStepAOT(_ForwardBackwardNextStepAOTBase):
    def __init__(self, k: int = 8, train_frac: float = 0.5, steps: int = 300, lr: float = 1e-3, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, standardize: bool = True, max_samples: int | None = None, seed: int = 0):
        super().__init__(k, train_frac, standardize, max_samples, seed)
        self.steps, self.lr, self.d_model, self.n_heads, self.n_layers = int(steps), float(lr), int(d_model), int(n_heads), int(n_layers)

    def fit_r2(self, ctx_tr, y_tr, ctx_te, y_te):
        _, k, D = ctx_tr.shape
        model = TinyTransformerRegressor(D, D, k, self.d_model, self.n_heads, self.n_layers).to(ctx_tr.device)
        train(model, ctx_tr.float(), y_tr.float(), loss="mse", steps=self.steps, lr=self.lr)
        with torch.no_grad():
            pred = model(ctx_te.float())
        return r2_vec(y_te, pred, y_tr.mean(dim=0, keepdim=True))
