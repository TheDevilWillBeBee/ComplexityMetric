from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Optional

import numpy as np
import torch

from DynamicalSystems import Rollout


@dataclass(frozen=True, eq=False)
class Embedding(Rollout):
    def __post_init__(self):
        if self.tensor.ndim != 3:
            raise ValueError("Embedding tensor must have shape (B,T,D)")
        object.__setattr__(self, "spatial_dim", 0)

    @property
    def D(self) -> int:
        return int(self.tensor.shape[-1])

    @torch.no_grad()
    def visualize(
        self,
        method: str = "pca",
        n_components: int = 2,
        *,
        connect: bool = True,
        max_cols: Optional[int] = None,
        title: Optional[str] = None,
        **kwargs: Any,
    ):
        if n_components not in (2, 3):
            raise ValueError("n_components must be 2 or 3")

        z = self.tensor.detach().float()
        B, T, _ = z.shape
        ys = [_reduce(z[b], method, n_components, **kwargs) for b in range(B)]

        import matplotlib.pyplot as plt
        from PIL import Image

        ncols = B if max_cols is None else max(1, min(int(max_cols), B))
        nrows = int(np.ceil(B / ncols))
        fig = plt.figure(figsize=(3 * ncols, 3 * nrows), dpi=150)
        if title:
            fig.suptitle(title)

        t = np.arange(T)
        for i, y in enumerate(ys):
            ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d" if n_components == 3 else None)
            if connect:
                ax.plot(*y.T, alpha=0.35)
            ax.scatter(*y.T, c=t, s=10)
            ax.scatter(*y[0], marker="o", s=40)
            ax.scatter(*y[-1], marker="x", s=40)
            ax.set_xticks([])
            ax.set_yticks([])
            if n_components == 3:
                ax.set_zticks([])

        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def _reduce(z: torch.Tensor, method: str, n_components: int, **kwargs) -> np.ndarray:
    method = method.lower()
    if method == "pca":
        z = z - z.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(z, full_matrices=False)
        k = min(n_components, vh.shape[0])
        y = z @ vh[:k].T
        if k < n_components:
            y = torch.cat([y, torch.zeros(z.shape[0], n_components - k, device=z.device)], dim=1)
        return y.cpu().numpy()

    z_np = z.cpu().numpy()
    if method in ("tsne", "t-sne"):
        from sklearn.manifold import TSNE

        if len(z_np) < 2:
            raise ValueError("t-SNE needs at least two time points")
        perplexity = min(30, max(1, (len(z_np) - 1) // 3), len(z_np) - 1)
        return TSNE(
            n_components=n_components,
            init=kwargs.pop("init", "pca"),
            learning_rate=kwargs.pop("learning_rate", "auto"),
            perplexity=kwargs.pop("perplexity", perplexity),
            random_state=kwargs.pop("random_state", 0),
            **kwargs,
        ).fit_transform(z_np)

    if method == "umap":
        import umap

        return umap.UMAP(
            n_components=n_components,
            random_state=kwargs.pop("random_state", 0),
            **kwargs,
        ).fit_transform(z_np)

    raise ValueError("method must be 'pca', 'tsne', or 'umap'")
