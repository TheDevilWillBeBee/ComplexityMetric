from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn.functional as F
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter


class VideoWriter:
    def __init__(self, filename: str = "_autoplay.mp4", fps: float = 30.0, **kw):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in (np.float32, np.float64):
            img = np.uint8(img.clip(0, 1) * 255)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.writer.write_frame(img)

    def close(self):
        if self.writer:
            self.writer.close()
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()
        if self.params["filename"] == "_autoplay.mp4":
            self.show()

    def show(self, **kw):
        from IPython.display import display
        import moviepy.editor as mvp

        self.close()
        display(mvp.ipython_display(self.params["filename"], **kw))


def depthwise_conv1d(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """x:(B,C,W), filters:(F,K) -> (B,C*F,W) with circular padding."""
    B, C, W = x.shape
    filt, K = filters.shape
    y = x.reshape(B * C, 1, W)
    y = F.pad(y, ((K - 1) // 2, K // 2), mode="circular")
    y = F.conv1d(y, filters[:, None])
    return y.reshape(B, C * filt, W)


def depthwise_conv2d(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    """x:(B,C,H,W), filters:(F,KH,KW) -> (B,C*F,H,W) with circular padding."""
    B, C, H, W = x.shape
    filt, KH, KW = filters.shape
    y = x.reshape(B * C, 1, H, W)
    y = F.pad(y, ((KW - 1) // 2, KW // 2, (KH - 1) // 2, KH // 2), mode="circular")
    y = F.conv2d(y, filters[:, None])
    return y.reshape(B, C * filt, H, W)


def batched_lookup(table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """table:(B,N), idx:(B,...) long -> values with shape idx."""
    B = table.shape[0]
    flat = idx.view(B, -1)
    out = table.gather(1, flat)
    return out.view_as(idx)


def circular_conv2d_fft(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Circular convolution via FFT. x:(B,C,H,W), k:(B,C,KH,KW) -> (B,C,H,W)."""
    B, C, H, W = x.shape
    _, Ck, KH, KW = k.shape
    if Ck != C:
        raise ValueError("k must have same channel count as x")
    kp = torch.zeros((B, C, H, W), device=x.device, dtype=x.dtype)
    kp[:, :, :KH, :KW] = k
    kp = torch.roll(kp, shifts=(-KH // 2, -KW // 2), dims=(-2, -1))
    return torch.real(torch.fft.ifft2(torch.fft.fft2(x) * torch.fft.fft2(kp)))


def parse_bs(desc: str):
    s = desc.strip().upper().replace(" ", "")
    mB = re.search(r"B([^S]*)", s)
    mS = re.search(r"S(.*)", s)

    def nums(part: str):
        if any(ch in part for ch in ",;:_-"):
            return {int(x) for x in re.findall(r"\d+", part)}
        return {int(ch) for ch in part if ch.isdigit()}

    Bset = nums(mB.group(1)) if mB else set()
    Sset = nums(mS.group(1)) if mS else set()
    return Bset, Sset


def bs_tables(Bset, Sset, L: int, B: int, device):
    Bt = torch.zeros(B, L, device=device)
    St = torch.zeros(B, L, device=device)
    for n in Bset:
        if 0 <= n < L:
            Bt[:, n] = 1.0
    for n in Sset:
        if 0 <= n < L:
            St[:, n] = 1.0
    return Bt, St


def fractal_noise_2d(
    B, C, H, W, *, octaves=4, persistence=0.4, black_prop=0.25, device="cpu"
):
    """Cheap perlin-ish fractal noise using random grids + bilinear upsampling."""
    device = torch.device(device)
    out = 0.0
    norm = 0.0
    for i in range(octaves):
        amp = persistence ** (i + 1)
        h0 = max(1, H // (2**i))
        w0 = max(1, W // (2**i))
        n = torch.rand(B, C, h0, w0, device=device)
        n = F.interpolate(n, size=(H, W), mode="bilinear", align_corners=False)
        out = out + amp * n
        norm = norm + amp
    out = out / (norm + 1e-8)
    return ((out - black_prop) / (1.0 - black_prop + 1e-8)).clamp(0.0, 1.0)


def as_tensor(x, device, dtype=torch.float32):
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


_parse_bs = parse_bs
_bs_tables = bs_tables
_as_tensor = as_tensor
