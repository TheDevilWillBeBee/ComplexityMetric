from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from DynamicalSystems import Rollout
from .embedding import Embedding


class FeatureExtractor(nn.Module):
    input_type = "continuous"
    output_type = "continuous"

    def __init__(self, spatial_dim: int, *, unit_norm: bool = False, device: str | torch.device = "cpu"):
        super().__init__()
        if spatial_dim not in (1, 2):
            raise ValueError("spatial_dim must be 1 or 2")
        self.spatial_dim = int(spatial_dim)
        self.unit_norm = bool(unit_norm)
        self.device = torch.device(device)

    def forward(self, rollout: Rollout, *, chunk: int = 0) -> Embedding:
        if not isinstance(rollout, Rollout):
            raise TypeError("FeatureExtractor input must be a Rollout")
        if rollout.spatial_dim != self.spatial_dim:
            raise ValueError(f"expected spatial_dim={self.spatial_dim}, got {rollout.spatial_dim}")

        rollout = self._typed(rollout)
        x = rollout.tensor.to(self.device)
        B, T = x.shape[:2]
        frames = x.reshape(B * T, *x.shape[2:])
        self._build(frames.shape[1])

        if chunk and chunk < frames.shape[0]:
            z = torch.cat([self.embed(frames[i : i + chunk]) for i in range(0, frames.shape[0], chunk)])
        else:
            z = self.embed(frames)
        z = z.reshape(B, T, -1)
        if self.unit_norm:
            z = F.normalize(z, dim=-1)

        return Embedding(
            tensor=z,
            steps=rollout.steps,
            every=rollout.every,
            skip=rollout.skip,
            is_discrete=self.output_type == "discrete",
            num_states=rollout.num_states if self.output_type == "discrete" else None,
            system_name=rollout.system_name,
        )

    def _typed(self, rollout: Rollout) -> Rollout:
        if self.input_type == "continuous":
            return rollout.to_continuous()
        if self.input_type == "discrete":
            return rollout.to_discrete()
        return rollout

    def _build(self, channels: int):
        pass

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class DiscreteFlatten(FeatureExtractor):
    input_type = "discrete"
    output_type = "discrete"

    def __init__(self, spatial_dim: int, *, device: str | torch.device = "cpu"):
        super().__init__(spatial_dim, unit_norm=False, device=device)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)


class ContinuousFlatten(FeatureExtractor):
    input_type = "continuous"
    output_type = "continuous"

    def __init__(self, spatial_dim: int, *, device: str | torch.device = "cpu"):
        super().__init__(spatial_dim, unit_norm=False, device=device)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(x.shape[0], -1)


class Flatten(DiscreteFlatten):
    pass


class SpatialStatistics(FeatureExtractor):
    input_type = "continuous"

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, C, W) for 1D or (B*T, C, H, W) for 2D.
        # returns: (B*T, 2*C) = [mean per channel, std per channel].
        dims = _spatial_dims(x)
        x = x.float()
        return torch.cat((x.mean(dim=dims), x.std(dim=dims, unbiased=False)), dim=1)


class RandomConvNet(FeatureExtractor):
    def __init__(
        self,
        spatial_dim: int,
        *,
        embed_dim: int = 128,
        base_channels: int = 32,
        num_stages: int = 3,
        layers_per_stage: int | Sequence[int] = 2,
        activation: str = "relu",
        reduction: str = "gram", # 'avg' | 'max' | 'gram'
        norm: str = "none", # 'none' | 'instance' | 'group' | 'layer'
        use_pooling: bool = False,
        kernel_size: int = 3,
        kernel_growth: int = 2,
        unit_norm: bool = True,
        seed: Optional[int] = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__(spatial_dim, unit_norm=unit_norm, device=device)
        if reduction not in ("avg", "max", "gram"):
            raise ValueError("reduction must be 'avg', 'max', or 'gram'")
        self.embed_dim = int(embed_dim)
        self.base_channels = int(base_channels)
        self.num_stages = int(num_stages)
        self.layers_per_stage = _layers_per_stage(layers_per_stage, self.num_stages)
        self.activation = activation
        self.kernel_size = int(kernel_size)
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.kernel_growth = int(kernel_growth)
        self.reduction = reduction
        self.norm = norm
        self.use_pooling = bool(use_pooling)
        self.seed = seed
        self.net = None
        self.proj = None

    def _build(self, channels: int):
        if self.net is not None:
            return
        Conv, Pool = _conv(self.spatial_dim), _avg_pool(self.spatial_dim)
        with _seeded(self.seed):
            layers = []
            c = int(channels)
            out = self.base_channels
            for stage in range(self.num_stages):
                k = self.kernel_size if self.use_pooling else self.kernel_size + stage * self.kernel_growth
                if k % 2 == 0:
                    k += 1
                pad = k // 2
                for _ in range(self.layers_per_stage[stage]):
                    layers += [
                        Conv(c, out, k, padding=pad, padding_mode="circular"),
                        _norm(self.spatial_dim, self.norm, out),
                        _act(self.activation),
                    ]
                    c = out
                if self.use_pooling and stage < self.num_stages - 1:
                    layers.append(Pool(2, stride=2))
                out *= 2
            red_dim = c if self.reduction != "gram" else c * c
            self.net = nn.Sequential(*layers).to(self.device)
            self.proj = nn.Linear(red_dim, self.embed_dim).to(self.device)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        # Random features expect inputs scaled to [0, 1]; center them before convolution.
        h = self.net(x.float() - 0.5)
        z = _reduce_spatial(h, self.reduction)
        return self.proj(z)


class RandomVGG(FeatureExtractor):
    def __init__(
        self,
        spatial_dim: int,
        *,
        embed_dim: int = 128,
        channels: Sequence[int] = (16, 32, 64, 128, 128),
        convs_per_block: int | Sequence[int] = (2, 2, 3, 3, 3),
        activation: str = "relu",
        unit_norm: bool = True,
        reduction: str = "gram", # 'avg' | 'max' | 'gram'
        taps: Optional[Sequence[int]] = None,
        tap_dim: Optional[int] = None,
        seed: Optional[int] = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__(spatial_dim, unit_norm=unit_norm, device=device)
        if reduction not in ("avg", "max", "gram"):
            raise ValueError("reduction must be 'avg', 'max', or 'gram'")
        self.embed_dim = int(embed_dim)
        self.channels = tuple(int(c) for c in channels)
        self.convs_per_block = _vgg_convs_per_block(convs_per_block, len(self.channels))
        self.activation = activation
        self.seed = seed
        self.reduction = reduction
        self.taps = _vgg_texture_taps(self.convs_per_block) if taps is None else tuple(sorted(int(t) for t in taps))
        self.tap_dim = self.embed_dim if tap_dim is None else int(tap_dim)
        self.net = None
        self.tap_positions = None
        self.tap_projs = None
        self.proj = None

    def _build(self, channels: int):
        if self.net is not None:
            return
        Conv, Pool = _conv(self.spatial_dim), _pool(self.spatial_dim)
        total_convs = sum(self.convs_per_block)
        if not self.taps:
            raise ValueError("taps must contain at least one conv-layer index")
        if len(set(self.taps)) != len(self.taps):
            raise ValueError("taps must not contain duplicate conv-layer indices")
        if any(t < 0 or t >= total_convs for t in self.taps):
            raise ValueError(f"taps must be in [0, {total_convs - 1}] for this RandomVGG")
        with _seeded(self.seed):
            layers = []
            tap_positions = []
            tap_dims = []
            c = int(channels)
            conv_idx = 0
            for out, depth in zip(self.channels, self.convs_per_block):
                for _ in range(depth):
                    layers.append(Conv(c, out, 3, padding=1, padding_mode="circular"))
                    layers.append(_act(self.activation))
                    if conv_idx in self.taps:
                        tap_positions.append(len(layers) - 1)
                        tap_dims.append(out if self.reduction != "gram" else out * out)
                    conv_idx += 1
                    c = out
                layers.append(Pool(2))
            self.net = nn.Sequential(*layers).to(self.device)
            self.tap_positions = tuple(tap_positions)
            self.tap_projs = nn.ModuleList([nn.Linear(int(d), self.tap_dim) for d in tap_dims]).to(self.device)
            self.proj = nn.Linear(len(self.tap_positions) * self.tap_dim, self.embed_dim).to(self.device)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        # Random features expect inputs scaled to [0, 1]; center them before convolution.
        h = x.float() - 0.5
        zs = []
        tap_idx = 0
        for layer_idx, layer in enumerate(self.net):
            h = layer(h)
            if tap_idx < len(self.tap_positions) and layer_idx == self.tap_positions[tap_idx]:
                z = self.tap_projs[tap_idx](_reduce_spatial(h, self.reduction))
                zs.append(F.normalize(z, dim=-1))
                tap_idx += 1
        return self.proj(torch.cat(zs, dim=1))


class VGG16(FeatureExtractor):
    def __init__(
        self,
        *,
        embed_dim: Optional[int] = None,
        pretrained: bool = True,
        unit_norm: bool = True,
        seed: Optional[int] = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__(2, unit_norm=unit_norm, device=device)
        self.embed_dim = None if embed_dim is None else int(embed_dim)
        self.pretrained = bool(pretrained)
        self.seed = seed
        self.features = None
        self.proj = None

    def _build(self, channels: int):
        if self.features is not None:
            return
        del channels
        from torchvision.models import VGG16_Weights, vgg16

        weights = VGG16_Weights.IMAGENET1K_V1 if self.pretrained else None
        with _seeded(self.seed):
            model = vgg16(weights=weights)
            self.features = model.features.eval().to(self.device)
            self.proj = nn.Identity() if self.embed_dim is None else nn.Linear(512, self.embed_dim, bias=False).to(self.device)
        for p in self.features.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = _rgb_2d(x.float().clamp(0, 1))
        if self.pretrained:
            x = _imagenet_normalize(x)
        else:
            x = x - 0.5
        return self.proj(_reduce_spatial(self.features(x), "avg"))


class CLIP(FeatureExtractor):
    def __init__(
        self,
        *,
        embed_dim: Optional[int] = None,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        image_size: int = 224,
        unit_norm: bool = True,
        device: str | torch.device = "cpu",
    ):
        super().__init__(2, unit_norm=unit_norm, device=device)
        import open_clip

        self.image_size = int(image_size)
        self.model = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)[0].eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        native_dim = int(getattr(self.model.visual, "output_dim", 512))
        self.proj = nn.Identity() if embed_dim is None else nn.Linear(native_dim, int(embed_dim), bias=False).to(self.device)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().clamp(0, 1)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 2:
            x = torch.cat([x, x[:, :1]], dim=1)
        else:
            x = x[:, :3]
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, (self.image_size, self.image_size), mode="bilinear", align_corners=False)
        mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device=x.device).view(1, 3, 1, 1)
        std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=x.device).view(1, 3, 1, 1)
        return self.proj(self.model.encode_image((x - mean) / std).float())


def _conv(dim: int):
    return {1: nn.Conv1d, 2: nn.Conv2d}[dim]


def _pool(dim: int):
    return {1: nn.MaxPool1d, 2: nn.MaxPool2d}[dim]


def _avg_pool(dim: int):
    return {1: nn.AvgPool1d, 2: nn.AvgPool2d}[dim]


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
    raise ValueError("activation must be relu, gelu, silu, swish, or tanh")


def _norm(dim: int, kind: str, channels: int) -> nn.Module:
    kind = kind.lower()
    if kind == "none":
        return nn.Identity()
    if kind == "instance":
        return {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d}[dim](channels, affine=True)
    if kind == "group":
        return nn.GroupNorm(_pick_groups(channels), channels)
    if kind == "layer":
        return nn.GroupNorm(1, channels)
    raise ValueError("norm must be none, instance, group, or layer")


def _pick_groups(channels: int, max_groups: int = 8) -> int:
    groups = min(int(max_groups), int(channels))
    while groups > 1 and channels % groups:
        groups -= 1
    return groups


def _layers_per_stage(value: int | Sequence[int], num_stages: int) -> list[int]:
    if isinstance(value, int):
        return [int(value)] * int(num_stages)
    layers = [int(v) for v in value]
    if len(layers) != int(num_stages):
        raise ValueError("layers_per_stage must be an int or one value per stage")
    return layers


def _vgg_convs_per_block(value: int | Sequence[int], num_blocks: int) -> list[int]:
    if isinstance(value, int):
        return [int(value)] * int(num_blocks)
    depths = [int(v) for v in value]
    if len(depths) < int(num_blocks):
        raise ValueError("convs_per_block must be an int or have at least one value per VGG block")
    return depths[: int(num_blocks)]


def _vgg_texture_taps(convs_per_block: Sequence[int]) -> tuple[int, ...]:
    taps = []
    start = 0
    for depth in convs_per_block:
        taps.append(start)
        start += int(depth)
    return tuple(taps)


def _rgb_2d(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    if x.shape[1] == 2:
        return torch.cat([x, x[:, :1]], dim=1)
    return x[:, :3]


def _imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor((0.485, 0.456, 0.406), device=x.device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def _spatial_dims(x: torch.Tensor):
    return tuple(range(2, x.ndim))


def _reduce_spatial(x: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "avg":
        return x.mean(dim=_spatial_dims(x))
    if reduction == "max":
        return x.amax(dim=_spatial_dims(x))
    n, c = x.shape[:2]
    f = x.reshape(n, c, -1)
    return ((f @ f.transpose(1, 2)) / f.shape[-1]).reshape(n, c * c)


@contextmanager
def _seeded(seed: Optional[int]):
    if seed is None:
        yield
        return
    state = torch.get_rng_state()
    torch.manual_seed(int(seed))
    try:
        yield
    finally:
        torch.set_rng_state(state)
