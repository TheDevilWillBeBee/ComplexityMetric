# aot_models.py
from __future__ import annotations
import torch
import torch.nn as nn


def _act(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unknown activation: {name}")


class MLP(nn.Module):
    """
    Tiny MLP: (N,in_dim)->(N,out_dim)
    """
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, depth: int = 2, act: str = "relu"):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        layers = []
        d = in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), _act(act)]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyTransformerClassifier(nn.Module):
    """
    Sequence classifier: (N,L,in_dim)->(N,out_dim)
    Uses a learned [CLS] token + TransformerEncoder.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,L,in_dim)
        h = self.proj(x)
        cls = self.cls.expand(h.size(0), 1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.enc(h)
        return self.head(h[:, 0])  # CLS


class TinyTransformerRegressor(nn.Module):
    """
    Sequence -> vector regressor:
      x: (N,L,in_dim) -> (N,out_dim)
    Minimal transformer with learned positional embeddings.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        seq_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.seq_len, d_model))  # learned positions
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,L,in_dim)
        if x.size(1) != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {x.size(1)}")
        h = self.proj(x) + self.pos
        h = self.enc(h)
        return self.head(h[:, -1])  # use last token representation