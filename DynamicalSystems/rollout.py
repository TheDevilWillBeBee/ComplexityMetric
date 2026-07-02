from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .utils import VideoWriter


ToRGBFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True, eq=False)
class Rollout:
    tensor: torch.Tensor
    steps: Optional[int] = None
    every: int = 1
    skip: int = 0
    is_discrete: bool = False
    num_states: Optional[int | Sequence[int]] = None
    spatial_dim: Optional[int] = None
    to_rgb_fn: Optional[ToRGBFn] = None
    system_name: Optional[str] = None

    def __post_init__(self):
        if self.tensor.ndim < 3:
            raise ValueError("Rollout tensor must have shape (B,T,C,...)")
        if self.spatial_dim is None:
            object.__setattr__(self, "spatial_dim", max(0, self.tensor.ndim - 3))

    @property
    def shape(self):
        return self.tensor.shape

    @property
    def ndim(self) -> int:
        return self.tensor.ndim

    @property
    def dtype(self):
        return self.tensor.dtype

    @property
    def device(self):
        return self.tensor.device

    @property
    def T(self) -> int:
        return int(self.tensor.shape[1])

    @property
    def channels(self) -> int:
        return int(self.tensor.shape[2])

    def as_tensor(self) -> torch.Tensor:
        return self.tensor

    def with_tensor(self, tensor: torch.Tensor, **metadata) -> "Rollout":
        return replace(self, tensor=tensor, **metadata)

    def __getitem__(self, item: int | slice) -> "Rollout":
        if isinstance(item, int):
            start = item + self.T if item < 0 else item
            if start < 0 or start >= self.T:
                raise IndexError("Rollout temporal index out of range")
            item = slice(start, start + 1)
        elif not isinstance(item, slice):
            raise TypeError("Rollout indices must be integers or slices over the temporal dimension")

        tensor = self.tensor[:, item]
        return self.with_tensor(tensor, steps=tensor.shape[1])

    def to(self, *args, **kwargs) -> "Rollout":
        return self.with_tensor(self.tensor.to(*args, **kwargs))

    def cpu(self) -> "Rollout":
        return self.to("cpu")

    def detach(self) -> "Rollout":
        return self.with_tensor(self.tensor.detach())

    def clone(self) -> "Rollout":
        return self.with_tensor(self.tensor.clone())

    def to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        if self.to_rgb_fn is not None:
            return self.to_rgb_fn(x)
        return default_to_rgb(x)

    def to_continuous(
        self,
        *,
        method: str = "onehot",
        dtype: torch.dtype = torch.float32,
    ) -> "Rollout":
        if not self.is_discrete:
            return self.with_tensor(self.tensor.to(dtype=dtype))
        if method != "onehot":
            raise ValueError("Only method='onehot' is supported for now")

        x = self.tensor.long()
        states = _states_per_channel(self.num_states, self.channels)
        parts = []
        for c, K in enumerate(states):
            xc = x[:, :, c].clamp(0, K - 1)
            oh = F.one_hot(xc, num_classes=K).to(dtype=dtype)
            parts.append(oh.movedim(-1, 2))
        y = torch.cat(parts, dim=2)
        return self.with_tensor(y, is_discrete=False, num_states=None)

    def to_discrete(
        self,
        *,
        num_bins: int = 2,
        value_range: tuple[float, float] = (0.0, 1.0),
    ) -> "Rollout":
        if self.is_discrete:
            return self
        lo, hi = map(float, value_range)
        if hi <= lo:
            raise ValueError("value_range must be (lo, hi) with hi > lo")
        x = ((self.tensor.float() - lo) / (hi - lo)).clamp(0.0, 1.0)
        y = (x * int(num_bins)).floor().long().clamp(0, int(num_bins) - 1)
        return self.with_tensor(y, is_discrete=True, num_states=int(num_bins))

    @torch.no_grad()
    def visualize(
        self,
        *,
        t: int = -1,
        kind: Optional[str] = None,
        normalize: bool = False,
    ):
        from PIL import Image

        y = self.tensor
        if kind is None:
            kind = self._infer_visual_kind(y)

        if kind == "1d_rollout":
            B, T, C, W = y.shape
            rgb = self.to_rgb(y.reshape(B * T, C, W)).float()
            if rgb.ndim == 4:
                rgb = rgb.mean(dim=2)
            if rgb.ndim != 3:
                raise ValueError("to_rgb for 1D rollout must return (B,3,W) or (B,3,H,W)")
            rgb = rgb.reshape(B, T, rgb.shape[1], rgb.shape[2])
            if normalize:
                mn = rgb.amin(dim=(1, 2, 3), keepdim=True)
                mx = rgb.amax(dim=(1, 2, 3), keepdim=True)
                rgb = (rgb - mn) / (mx - mn + 1e-8)
            else:
                rgb = rgb.clamp(0, 1)
            img = rgb.permute(0, 1, 3, 2)
            img = (img * 255).byte().cpu().numpy()
            return Image.fromarray(np.concatenate([img[b] for b in range(B)], axis=1), mode="RGB")

        if kind == "2d_rollout":
            x = y[:, t]
        elif kind in ("2d_state", "1d_state"):
            x = y[:, 0] if y.ndim in (4, 5) and y.shape[1] == 1 else y
        else:
            raise ValueError(f"Unknown kind: {kind}")

        rgb = self.to_rgb(x).float()
        if normalize:
            mn = rgb.amin(dim=(1, 2, 3), keepdim=True)
            mx = rgb.amax(dim=(1, 2, 3), keepdim=True)
            rgb = (rgb - mn) / (mx - mn + 1e-8)
        else:
            rgb = rgb.clamp(0, 1)

        arr = (rgb.clamp(0, 1) * 255).byte().cpu().permute(0, 2, 3, 1).numpy()
        return Image.fromarray(np.concatenate([arr[b] for b in range(arr.shape[0])], axis=1), mode="RGB")

    @torch.no_grad()
    def visualize_video(
        self,
        *,
        filename: str = "_autoplay.mp4",
        fps: float = 30.0,
        every: int = 1,
        t0: int = 0,
        t1: Optional[int] = None,
        kind: Optional[str] = None,
        normalize: bool = False,
    ):
        y = self.tensor
        if kind is None:
            kind = self._infer_visual_kind(y)
        if kind in ("1d_rollout", "1d_state"):
            return NotImplemented

        frames = y if kind == "2d_rollout" else y.unsqueeze(1)
        _, T, _, _, _ = frames.shape
        t1 = T if t1 is None else min(int(t1), T)
        every = max(1, int(every))

        with VideoWriter(filename=filename, fps=fps) as vw:
            for i in range(int(t0), t1, every):
                rgb = self.to_rgb(frames[:, i]).float()
                if normalize:
                    mn = rgb.amin(dim=(1, 2, 3), keepdim=True)
                    mx = rgb.amax(dim=(1, 2, 3), keepdim=True)
                    rgb = (rgb - mn) / (mx - mn + 1e-8)
                else:
                    rgb = rgb.clamp(0, 1)
                arr = (rgb * 255).byte().cpu().permute(0, 2, 3, 1).numpy()
                vw.add(np.concatenate([arr[b] for b in range(arr.shape[0])], axis=1))
        return filename

    @staticmethod
    def _infer_visual_kind(y: torch.Tensor) -> str:
        if y.ndim == 5:
            return "2d_rollout"
        if y.ndim == 3:
            return "1d_state"
        if y.ndim == 4:
            if y.shape[1] > 8 and y.shape[2] <= 8:
                return "1d_rollout"
            if y.shape[1] <= 8 and y.shape[2] > 8:
                return "2d_state"
            return "1d_rollout"
        raise ValueError(f"Unsupported tensor rank: {y.ndim}")


def default_to_rgb(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(-2)
    if x.ndim != 4:
        raise ValueError("to_rgb expects (B,C,W) or (B,C,H,W)")
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    if x.shape[1] >= 3:
        return x[:, :3]
    g = x.mean(dim=1, keepdim=True)
    return g.repeat(1, 3, 1, 1)


def _states_per_channel(num_states, channels: int) -> list[int]:
    if num_states is None:
        raise ValueError("num_states is required for one-hot conversion")
    if isinstance(num_states, int):
        return [int(num_states)] * channels
    states = [int(k) for k in num_states]
    if len(states) == 1:
        return states * channels
    if len(states) != channels:
        raise ValueError("num_states must be an int or one value per channel")
    return states
