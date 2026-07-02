from __future__ import annotations

from typing import Callable, Dict, Iterator, Optional
import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from .base import DynamicalSystem
from .utils import (
    _as_tensor,
    _bs_tables,
    _parse_bs,
    batched_lookup,
    depthwise_conv1d,
    depthwise_conv2d,
    fractal_noise_2d,
)


class NoiseDynamics(DynamicalSystem):
    """
    Produces i.i.d. noise each step.

    - discrete=True: samples {0,1}
    - discrete=False: samples in [0,1] with optional EMA smoothing:
        x_next = (1 - ema) * noise + ema * x
    """

    def __init__(
        self,
        channels: int = 1,
        discrete: bool = True,
        ema: float = 0.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__(device=device)
        self.C = int(channels)
        self.discrete = bool(discrete)
        self.ema = float(ema)

    @staticmethod
    def sample_params(
        B: int, device: str | torch.device = "cpu", **kwargs
    ) -> Dict[str, torch.Tensor]:
        _ = (B, device, kwargs)
        return {}

    @staticmethod
    def iter_params(
        *, device: str | torch.device = "cpu", batch_size: int = 64, **kwargs
    ) -> Iterator[Dict[str, torch.Tensor]]:
        _ = (device, batch_size, kwargs)
        yield {}

    def seed(
        self,
        B: int = 1,
        W: int = 128,
        H: Optional[int] = None,
        generator: Optional[Callable[..., torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        if generator is not None:
            return generator(B=B, C=self.C, H=H, W=W, device=self.device, **kwargs).to(
                self.device
            )

        if H is None:
            shape = (B, self.C, W)
        else:
            shape = (B, self.C, H, W)

        if self.discrete:
            return (torch.rand(*shape, device=self.device) < 0.5).float()
        return torch.rand(*shape, device=self.device)

    def forward(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        _ = params
        if self.discrete:
            return (torch.rand_like(x, dtype=torch.float32) < 0.5).float()

        noise = torch.rand_like(x, dtype=torch.float32)
        blend = max(0.0, min(1.0, float(self.ema)))
        if blend <= 0.0:
            return noise
        return (1.0 - blend) * noise + blend * x.to(dtype=torch.float32)

    def to_integer(self, x: torch.Tensor, num_bins: int = 2) -> torch.Tensor:
        if self.discrete:
            return (x > 0.5).long()
        # Continuous noise in [0,1]: quantize into num_bins uniform bins
        x_f = x.float().clamp(0.0, 1.0)
        return (x_f * num_bins).floor().long().clamp(0, num_bins - 1)


# -----------------------------
# 1D Binary CA
# -----------------------------


class BinaryCA1D(DynamicalSystem):
    """
    State:  (B,1,W)  in {0,1}
    Params: {'rule': (B,2**K)}  where rule[idx] gives next cell value
    """

    discrete_state = True
    num_states_per_channel = 2

    def __init__(self, kernel_size: int = 3, device: str | torch.device = "cpu"):
        super().__init__(device=device)
        self.K = int(kernel_size)
        self.N = 1 << self.K
        w = torch.tensor(
            [1 << p for p in reversed(range(self.K))], device=self.device
        ).float()
        self.register_buffer("kernel", w.view(1, self.K), persistent=False)

    @staticmethod
    def sample_params(
        B: int,
        device: str | torch.device = "cpu",
        p: float = 0.5,
        kernel_size: int = 3,
        rule_int: Optional[int] = None,
    ):
        """If rule_int is set and K=3, this matches the standard ECA numbering."""
        K = int(kernel_size)
        N = 1 << K
        device = torch.device(device)
        if rule_int is None:
            rule = (torch.rand(B, N, device=device) < p).float()
        else:
            r = torch.full((B,), int(rule_int), device=device, dtype=torch.long)
            rule = ((r[:, None] >> torch.arange(N, device=device)) & 1).float()
        return {"rule": rule}

    @staticmethod
    def iter_params(
        *,
        device: str | torch.device = "cpu",
        batch_size: int = 64,
        kernel_size: int = 3,
    ):
        """Iterate over all 256 elementary CA rules as batches."""
        K = int(kernel_size)
        if K not in [3, 4, 5]:
            raise ValueError("iter_params is only implemented for kernel_size=3, 4, or 5.")
        device = torch.device(device)
        N = 1 << K
        for start in range(0, 1 << N, batch_size):
            r = torch.arange(
                start, min(start + batch_size, 1 << N), device=device, dtype=torch.long
            )
            rule = ((r[:, None] >> torch.arange(N, device=device)) & 1).float()
            yield {"rule": rule}

    def seed(
        self,
        B: int = 1,
        W: int = 128,
        mode: str = "noise",
        p: float = 0.5,
        generator=None,
    ) -> torch.Tensor:
        if generator is not None:
            return generator(B=B, W=W, device=self.device, mode=mode, p=p).to(
                self.device
            )
        if mode == "noise":
            return (torch.rand(B, 1, W, device=self.device) < p).float()
        if mode == "zeros":
            return torch.zeros(B, 1, W, device=self.device)
        if mode == "ones":
            return torch.ones(B, 1, W, device=self.device)
        if mode == "single":
            x = torch.zeros(B, 1, W, device=self.device)
            x[:, 0, W // 2] = 1.0
            return x
        raise ValueError(f"unknown mode: {mode}")

    def forward(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        rule = params["rule"].to(device=x.device, dtype=torch.float32)
        if rule.ndim == 1:
            rule = rule[None, :].expand(x.shape[0], -1)
        idx = (
            depthwise_conv1d(x.float(), self.kernel).round().long().squeeze(1)
        )  # (B,W)
        return batched_lookup(rule, idx).unsqueeze(1)  # (B,1,W)

    @staticmethod
    def lambda_min(params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Return min(lambda_0, lambda_1) for the rule table.
        lambda_1 is the fraction of neighborhoods mapped to 1.
        """
        rule = params["rule"].float()
        if rule.ndim == 1:
            rule = rule[None, :]
        lam1 = rule.mean(dim=1)
        return torch.minimum(lam1, 1.0 - lam1)

    def to_integer(self, x: torch.Tensor, num_bins: int = 2) -> torch.Tensor:
        return (x > 0.5).long()

    def to_rgb(self, s: torch.Tensor) -> torch.Tensor:
        # simple binary viz
        s = s * 0.5 + 0.25
        if s.ndim == 3:
            return s[:, :1].repeat(1, 3, 1)
        return super().to_rgb(s)


# ---------------------------------------
# Outer-totalistic 1D CA
# ---------------------------------------


class OuterTotalisticCA1D(DynamicalSystem):
    """
    Binary 1D CA with odd kernel_size K.

    Neighbor sum excludes the center cell -> sum in [0 .. K-1].
    Rule represented by two tables:
      params["B"]: (B,K)  birth[sum]
      params["S"]: (B,K)  survive[sum]
    """

    discrete_state = True
    num_states_per_channel = 2

    def __init__(self, kernel_size: int = 7, device: str | torch.device = "cpu"):
        super().__init__(device=device)
        self.K = int(kernel_size)
        if self.K <= 0 or (self.K % 2) == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.pad = self.K // 2
        self.L = self.K  # sums: 0..K-1

        ones = torch.ones(1, 1, self.K, device=self.device)
        self.register_buffer("ones", ones, persistent=False)

    @staticmethod
    def sample_params(
        B: int,
        device: str | torch.device = "cpu",
        kernel_size: int = 7,
        p_birth: float = 0.5,
        p_survive: float = 0.5,
    ):
        K = int(kernel_size)
        if (K % 2) == 0:
            raise ValueError("kernel_size must be odd")
        dev = torch.device(device)
        return {
            "B": (torch.rand(B, K, device=dev) < p_birth).float(),
            "S": (torch.rand(B, K, device=dev) < p_survive).float(),
        }

    @staticmethod
    def from_desc(
        desc: str,
        *,
        B: int = 1,
        kernel_size: int = 7,
        device: str | torch.device = "cpu",
    ):
        """Build params dict from a B/S description like 'B1/S23'."""
        K = int(kernel_size)
        if (K % 2) == 0:
            raise ValueError("kernel_size must be odd")
        L = K
        Bset, Sset = _parse_bs(desc)
        Bt, St = _bs_tables(Bset, Sset, L=L, B=B, device=torch.device(device))
        return {"B": Bt, "S": St}

    @staticmethod
    def iter_params(
        *,
        kernel_size: int = 7,
        device: str | torch.device = "cpu",
        batch_size: int = 256,
        max_total_bits: int = 26,
        limit: int | None = None,
    ):
        """
        Iterate over ALL outer-totalistic rules as batches.

        Total rules = 2^(2*K). This is feasible for modest K (e.g. K<=13 => 2^26 rules).
        Guard: if 2*K > max_total_bits -> raise (set max_total_bits=None to disable).
        limit: optionally stop after yielding `limit` rules total.
        """
        K = int(kernel_size)
        assert K > 3, "for K<=3 use iter_params from BinaryCA1D"
        if (K % 2) == 0:
            raise ValueError("kernel_size must be odd")
        L = K
        total_bits = 2 * L
        if max_total_bits is not None and total_bits > int(max_total_bits):
            raise ValueError(
                f"Rule space too large: 2^(2*K)=2^{total_bits}. Reduce K or increase max_total_bits."
            )
        total_rules = 1 << total_bits

        dev = torch.device(device)
        ar = torch.arange(L, device=dev, dtype=torch.long)
        low_mask = (1 << L) - 1

        emitted = 0
        for start in range(0, total_rules, batch_size):
            if limit is not None and emitted >= limit:
                return
            end = min(start + batch_size, total_rules)
            if limit is not None:
                end = min(end, start + (limit - emitted))

            r = torch.arange(start, end, device=dev, dtype=torch.long)  # (b,)
            bmask = r & low_mask
            smask = r >> L

            Bt = ((bmask[:, None] >> ar) & 1).float()  # (b,L)
            St = ((smask[:, None] >> ar) & 1).float()
            yield {"B": Bt, "S": St}
            emitted += end - start

    def seed(
        self,
        B: int = 1,
        W: int = 128,
        mode: str = "noise",
        p: float = 0.5,
        generator=None,
    ) -> torch.Tensor:
        if generator is not None:
            return generator(B=B, W=W, device=self.device, mode=mode, p=p).to(
                self.device
            )
        if mode == "noise":
            return (torch.rand(B, 1, W, device=self.device) < p).float()
        if mode == "zeros":
            return torch.zeros(B, 1, W, device=self.device)
        if mode == "ones":
            return torch.ones(B, 1, W, device=self.device)
        if mode == "single":
            x = torch.zeros(B, 1, W, device=self.device)
            x[:, 0, W // 2] = 1.0
            return x
        raise ValueError(f"unknown mode: {mode}")

    def forward(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        x = x.to(self.device).float()
        B = x.shape[0]

        x01 = (x > 0.5).float()

        Bt = params["B"].to(x.device).float()
        St = params["S"].to(x.device).float()
        if Bt.ndim == 1:
            Bt = Bt[None, :].expand(B, -1)
        if St.ndim == 1:
            St = St[None, :].expand(B, -1)

        tot = F.conv1d(
            F.pad(x01, (self.pad, self.pad), mode="circular"), self.ones
        )  # includes center
        nsum = (tot - x01).round().long().squeeze(1)  # (B,W), excludes center

        born = batched_lookup(Bt, nsum).unsqueeze(1)
        surv = batched_lookup(St, nsum).unsqueeze(1)
        return torch.where(x01 > 0.5, surv, born)

    def lambda_min(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Return min(lambda_0, lambda_1) for outer-totalistic rules.
        Accounts for multiplicity of neighborhoods per neighbor-sum.
        """
        Bt = params["B"].float()
        St = params["S"].float()
        if Bt.ndim == 1:
            Bt = Bt[None, :]
        if St.ndim == 1:
            St = St[None, :]

        N = self.K - 1  # number of neighbors excluding center
        counts = torch.tensor(
            [math.comb(N, n) for n in range(self.L)],
            device=Bt.device,
            dtype=Bt.dtype,
        )
        lam1 = ((Bt + St) * counts[None, :]).sum(dim=1) / float(2 ** (N + 1))
        return torch.minimum(lam1, 1.0 - lam1)

    @staticmethod
    def lambda_min_from_desc(
        desc: str, *, kernel_size: Optional[int] = None, max_sum: Optional[int] = None
    ) -> float:
        """
        Compute min(lambda_0, lambda_1) from a B/S description.
        Provide either kernel_size (K) or max_sum (N). For 1D: N=K-1.
        """
        if max_sum is None:
            if kernel_size is None:
                raise ValueError("Provide kernel_size or max_sum")
            K = int(kernel_size)
            N = K - 1
        else:
            N = int(max_sum)

        Bset, Sset = _parse_bs(desc)
        L = N + 1
        counts = [math.comb(N, n) for n in range(L)]
        total = float(2 ** (N + 1))
        lam1 = (
            sum(counts[n] for n in Bset if 0 <= n < L)
            + sum(counts[n] for n in Sset if 0 <= n < L)
        ) / total
        return min(lam1, 1.0 - lam1)

    def to_integer(self, x: torch.Tensor, num_bins: int = 2) -> torch.Tensor:
        return (x > 0.5).long()

    def to_rgb(self, s: torch.Tensor) -> torch.Tensor:
        # simple binary viz
        s = s * 0.5 + 0.25
        if s.ndim == 3:
            return s[:, :1].repeat(1, 3, 1)
        return super().to_rgb(s)


# -----------------------------
# 2D Binary CA (3x3)
# -----------------------------


class BinaryCA2D(DynamicalSystem):
    """
    State:  (B,1,H,W) in {0,1}
    Params: {'rule': (B,512)}  (3x3 neighborhood -> 9-bit index)
    """

    discrete_state = True
    num_states_per_channel = 2

    def __init__(self, device: str | torch.device = "cpu"):
        super().__init__(device=device)
        w = torch.tensor(
            [[256, 128, 64], [32, 16, 8], [4, 2, 1]], device=self.device
        ).float()
        self.register_buffer(
            "kernel", torch.flip(w, dims=(1,)).view(1, 3, 3), persistent=False
        )

    @staticmethod
    def sample_params(B: int, device: str | torch.device = "cpu", p: float = 0.5):
        device = torch.device(device)
        return {"rule": (torch.rand(B, 512, device=device) < p).float()}

    @staticmethod
    def bs_rule(desc: str, device: str | torch.device = "cpu") -> torch.Tensor:
        """
        Life-like outer-totalistic rule like 'B3/S23' (Conway's Life).
        Returns: (512,) float rule table.
        """
        import re

        s = desc.strip().upper().replace("/", "")
        m = re.fullmatch(r"B(\d*)(?:S(\d*))?", s)
        if not m:
            raise ValueError(f"bad rule: {desc}")
        Bset = {int(ch) for ch in m.group(1)} if m.group(1) else set()
        Sset = {int(ch) for ch in m.group(2)} if m.group(2) else set()

        bits = torch.zeros(512, device=device, dtype=torch.float32)
        for x in range(512):
            pattern = [(x >> i) & 1 for i in range(9)]
            c = pattern[4]
            nsum = sum(pattern) - c
            bits[x] = (
                1.0 if ((c == 0 and nsum in Bset) or (c == 1 and nsum in Sset)) else 0.0
            )
        return bits

    def seed(
        self,
        B: int = 1,
        H: int = 64,
        W: int = 64,
        mode: str = "noise",
        p: float = 0.5,
        generator=None,
    ) -> torch.Tensor:
        if generator is not None:
            return generator(B=B, H=H, W=W, device=self.device, mode=mode, p=p).to(
                self.device
            )
        if mode == "noise":
            return (torch.rand(B, 1, H, W, device=self.device) < p).float()
        if mode == "zeros":
            return torch.zeros(B, 1, H, W, device=self.device)
        if mode == "ones":
            return torch.ones(B, 1, H, W, device=self.device)
        if mode == "single":
            x = torch.zeros(B, 1, H, W, device=self.device)
            x[:, :, H // 2, W // 2] = 1.0
            return x
        raise ValueError(f"unknown mode: {mode}")

    def forward(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        rule = params["rule"].to(device=x.device, dtype=torch.float32)
        if rule.ndim == 1:
            rule = rule[None, :].expand(x.shape[0], -1)
        idx = (
            depthwise_conv2d(x.float(), self.kernel).round().long().squeeze(1)
        )  # (B,H,W)
        return batched_lookup(rule, idx).unsqueeze(1)  # (B,1,H,W)

    @staticmethod
    def lambda_min(params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Return min(lambda_0, lambda_1) for the rule table.
        lambda_1 is the fraction of neighborhoods mapped to 1.
        """
        rule = params["rule"].float()
        if rule.ndim == 1:
            rule = rule[None, :]
        lam1 = rule.mean(dim=1)
        return torch.minimum(lam1, 1.0 - lam1)

    def to_integer(self, x: torch.Tensor, num_bins: int = 2) -> torch.Tensor:
        return (x > 0.5).long()

    def to_rgb(self, s: torch.Tensor) -> torch.Tensor:
        # simple binary viz
        s = s * 0.5 + 0.25
        if s.ndim == 4:
            return s[:, :1].repeat(1, 3, 1, 1)
        return super().to_rgb(s)


# ---------------------------------------
# Noisy Binary CA (2D)
# ---------------------------------------


class NoisyBinaryCA2D(BinaryCA2D):
    """
    Stochastic extension of BinaryCA2D.

    Each step, every cell is updated with probability ``alpha`` (update
    probability).  When a cell *is* updated it becomes:
      - random noise with probability ``noise_prob``  (value 0 or 1 drawn
        with P(1) = ``noise_bias``), or
      - the deterministic CA output with probability ``1 - noise_prob``.

    Cells that are *not* selected for update keep their old value.

    State:  (B,1,H,W)  in {0,1}
    Params: same as BinaryCA2D  {'rule': (B,512)}
    """

    def __init__(
        self,
        noise_prob: float = 0.0,
        alpha: float = 1.0,
        noise_bias: float = 0.5,
        device: str | torch.device = "cpu",
    ):
        super().__init__(device=device)
        self.noise_prob = float(noise_prob)
        self.alpha = float(alpha)
        self.noise_bias = float(noise_bias)

    # ---------- rule helpers ----------

    @staticmethod
    def majority_rule(ids: list[int] | None = None) -> torch.Tensor:
        """
        Build a 512-entry rule table from a majority-vote rule.

        The 3×3 neighbourhood is flattened to 9 bits (same ordering as
        ``BinaryCA2D``).  ``ids`` selects which of those 9 positions
        participate in the majority vote.  If ``ids is None`` all 9
        positions are used.

        Returns a ``(512,)`` float tensor suitable for
        ``params['rule']``.
        """
        if ids is None:
            ids = list(range(9))
        if not all(0 <= i < 9 for i in ids):
            raise ValueError("ids must be in [0, 8]")

        bits = torch.zeros(512, dtype=torch.float32)
        threshold = len(ids) // 2
        for x in range(512):
            s = sum(((x >> i) & 1) for i in ids)
            bits[x] = 1.0 if s > threshold else 0.0
        return bits

    @staticmethod
    def majority_rule_from_str(desc: str) -> torch.Tensor:
        """
        Parse a comma-separated string of position indices (e.g.
        ``"0,1,3,5,7"`` or ``"all"``) and return the majority-vote
        rule table.
        """
        desc = desc.strip().lower()
        if desc in ("all", ""):
            ids = None
        else:
            ids = [int(s.strip()) for s in desc.split(",")]
        return NoisyBinaryCA2D.majority_rule(ids)

    # ---------- parameter sampling ----------

    @staticmethod
    def sample_params(
        B: int,
        device: str | torch.device = "cpu",
        p: float = 0.5,
        rule: torch.Tensor | None = None,
    ):
        """
        If *rule* (a (512,) tensor) is given every batch element gets
        the same rule; otherwise a random table is sampled.
        """
        device = torch.device(device)
        if rule is not None:
            r = rule.to(device=device, dtype=torch.float32)
            if r.ndim == 1:
                r = r[None, :].expand(B, -1).contiguous()
            return {"rule": r}
        return {"rule": (torch.rand(B, 512, device=device) < p).float()}

    # ---------- forward ----------

    def forward(
        self, x: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Deterministic CA step (parent)
        ca_out = super().forward(x, params)  # (B,1,H,W)

        # Noise: 0 or 1 with P(1) = noise_bias
        noise = (torch.rand_like(x) < self.noise_bias).float()

        # Where noise replaces CA: per-cell coin flip with prob noise_prob
        use_noise = (torch.rand_like(x) < self.noise_prob).float()
        updated_value = use_noise * noise + (1.0 - use_noise) * ca_out

        # Stochastic update mask: each cell updated with prob alpha
        update_mask = (torch.rand_like(x) < self.alpha).float()

        return update_mask * updated_value + (1.0 - update_mask) * x


# ---------------------------------------
# Outer-totalistic 2D CA
# ---------------------------------------


class OuterTotalisticCA2D(DynamicalSystem):
    """
    Binary 2D CA with odd kernel_size K (KxK neighborhood).

    Neighbor sum excludes center -> sum in [0 .. K*K-1].
    Rule represented by two tables:
      params["B"]: (B,K*K) birth[sum]
      params["S"]: (B,K*K) survive[sum]

    Note: Iterating ALL rules is only feasible for small K (K=3 => 2^18 = 262,144 rules).
    """

    discrete_state = True
    num_states_per_channel = 2

    def __init__(self, kernel_size: int = 3, device: str | torch.device = "cpu"):
        super().__init__(device=device)
        self.K = int(kernel_size)
        if self.K <= 0 or (self.K % 2) == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.pad = self.K // 2
        self.L = self.K * self.K  # sums: 0..K*K-1

        ones = torch.ones(1, 1, self.K, self.K, device=self.device)
        self.register_buffer("ones", ones, persistent=False)

    @staticmethod
    def sample_params(
        B: int,
        device: str | torch.device = "cpu",
        kernel_size: int = 3,
        p_birth: float = 0.5,
        p_survive: float = 0.5,
    ):
        K = int(kernel_size)
        if (K % 2) == 0:
            raise ValueError("kernel_size must be odd")
        L = K * K
        dev = torch.device(device)
        return {
            "B": (torch.rand(B, L, device=dev) < p_birth).float(),
            "S": (torch.rand(B, L, device=dev) < p_survive).float(),
        }

    @staticmethod
    def from_desc(
        desc: str,
        *,
        B: int = 1,
        kernel_size: int = 3,
        device: str | torch.device = "cpu",
    ):
        """Build params dict from a B/S description like 'B3/S23' (works for any odd K)."""
        K = int(kernel_size)
        if (K % 2) == 0:
            raise ValueError("kernel_size must be odd")
        L = K * K
        Bset, Sset = _parse_bs(desc)
        Bt, St = _bs_tables(Bset, Sset, L=L, B=B, device=torch.device(device))
        return {"B": Bt, "S": St}

    @staticmethod
    def iter_params(
        *,
        kernel_size: int = 3,
        device: str | torch.device = "cpu",
        batch_size: int = 256,
        max_total_bits: int = 26,
        limit: int | None = None,
    ):
        """
        Iterate over ALL outer-totalistic 2D rules as batches.

        Total rules = 2^(2*K*K). Feasible for K=3 (2^18).
        Guard: if 2*K*K > max_total_bits -> raise (set max_total_bits=None to disable).
        """
        K = int(kernel_size)
        if (K % 2) == 0:
            raise ValueError("kernel_size must be odd")
        L = K * K
        total_bits = 2 * L
        if max_total_bits is not None and total_bits > int(max_total_bits):
            raise ValueError(
                f"Rule space too large: 2^(2*K*K)=2^{total_bits}. Reduce K or increase max_total_bits."
            )
        total_rules = 1 << total_bits

        dev = torch.device(device)
        ar = torch.arange(L, device=dev, dtype=torch.long)
        low_mask = (1 << L) - 1

        emitted = 0
        for start in range(0, total_rules, batch_size):
            if limit is not None and emitted >= limit:
                return
            end = min(start + batch_size, total_rules)
            if limit is not None:
                end = min(end, start + (limit - emitted))

            r = torch.arange(start, end, device=dev, dtype=torch.long)
            bmask = r & low_mask
            smask = r >> L

            Bt = ((bmask[:, None] >> ar) & 1).float()  # (b,L)
            St = ((smask[:, None] >> ar) & 1).float()
            # print(Bt.shape, St.shape)
            yield {"B": Bt, "S": St}
            emitted += end - start

    def seed(
        self,
        B: int = 1,
        H: int = 64,
        W: int = 64,
        mode: str = "noise",
        p: float = 0.5,
        generator=None,
    ) -> torch.Tensor:
        if generator is not None:
            return generator(B=B, H=H, W=W, device=self.device, mode=mode, p=p).to(
                self.device
            )
        if mode == "noise":
            return (torch.rand(B, 1, H, W, device=self.device) < p).float()
        if mode == "zeros":
            return torch.zeros(B, 1, H, W, device=self.device)
        if mode == "ones":
            return torch.ones(B, 1, H, W, device=self.device)
        if mode == "single":
            x = torch.zeros(B, 1, H, W, device=self.device)
            x[:, :, H // 2, W // 2] = 1.0
            return x
        raise ValueError(f"unknown mode: {mode}")

    def forward(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        x = x.to(self.device).float()
        B = x.shape[0]

        x01 = (x > 0.5).float()

        Bt = params["B"].to(x.device).float()
        St = params["S"].to(x.device).float()
        if Bt.ndim == 1:
            Bt = Bt[None, :].expand(B, -1)
        if St.ndim == 1:
            St = St[None, :].expand(B, -1)

        tot = F.conv2d(
            F.pad(x01, (self.pad, self.pad, self.pad, self.pad), mode="circular"),
            self.ones,
        )  # includes center
        nsum = (tot - x01).round().long().squeeze(1)  # (B,H,W), excludes center

        born = batched_lookup(Bt, nsum).unsqueeze(1)
        surv = batched_lookup(St, nsum).unsqueeze(1)
        return torch.where(x01 > 0.5, surv, born)

    def lambda_min(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Return min(lambda_0, lambda_1) for outer-totalistic rules.
        Accounts for multiplicity of neighborhoods per neighbor-sum.
        """
        Bt = params["B"].float()
        St = params["S"].float()
        if Bt.ndim == 1:
            Bt = Bt[None, :]
        if St.ndim == 1:
            St = St[None, :]

        N = self.L - 1  # number of neighbors excluding center
        counts = torch.tensor(
            [math.comb(N, n) for n in range(self.L)],
            device=Bt.device,
            dtype=Bt.dtype,
        )
        lam1 = ((Bt + St) * counts[None, :]).sum(dim=1) / float(2 ** (N + 1))
        return torch.minimum(lam1, 1.0 - lam1)

    @staticmethod
    def lambda_min_from_desc(
        desc: str, *, kernel_size: Optional[int] = None, max_sum: Optional[int] = None
    ) -> float:
        """
        Compute min(lambda_0, lambda_1) from a B/S description.
        Provide either kernel_size (K) or max_sum (N). For 2D: N=K*K-1.
        """
        if max_sum is None:
            if kernel_size is None:
                raise ValueError("Provide kernel_size or max_sum")
            K = int(kernel_size)
            N = K * K - 1
        else:
            N = int(max_sum)

        Bset, Sset = _parse_bs(desc)
        L = N + 1
        counts = [math.comb(N, n) for n in range(L)]
        total = float(2 ** (N + 1))
        lam1 = (
            sum(counts[n] for n in Bset if 0 <= n < L)
            + sum(counts[n] for n in Sset if 0 <= n < L)
        ) / total
        return min(lam1, 1.0 - lam1)

    def to_integer(self, x: torch.Tensor, num_bins: int = 2) -> torch.Tensor:
        return (x > 0.5).long()

    def to_rgb(self, s: torch.Tensor) -> torch.Tensor:
        # simple binary viz
        s = s * 0.5 + 0.25
        if s.ndim == 4:
            return s[..., :1, :, :].repeat(1, 3, 1, 1)

        return super().to_rgb(s)


# -----------------------------
# Neural CA (2D)
# -----------------------------


class NeuralCA2D(DynamicalSystem):
    """
    State: (B,C,H,W)
    Params:
      W1:(B,HIDDEN,C*F), b1:(B,HIDDEN), W2:(B,C,HIDDEN), b2:(B,C)
    """

    def __init__(
        self,
        channels: int = 16,
        hidden: int = 128,
        fire_rate: float = 0.5,
        dt: float = 1.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__(device=device)
        self.C = int(channels)
        self.H = int(hidden)
        self.fire_rate = float(fire_rate)
        self.dt = float(dt)

        ident = torch.tensor(
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]], device=self.device
        ).float()
        sx = (
            torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=self.device
            ).float()
            / 8.0
        )
        sy = (
            torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=self.device
            ).float()
            / 8.0
        )
        lap = (
            torch.tensor(
                [[1, 2, 1], [2, -12, 2], [1, 2, 1]], device=self.device
            ).float()
            / 4.0
        )
        self.register_buffer(
            "filters", torch.stack([ident, sx, sy, lap], dim=0), persistent=False
        )  # (F,3,3)
        self.F = 4
        self.in_dim = self.C * self.F

    @staticmethod
    def sample_params(
        B: int,
        device: str | torch.device = "cpu",
        channels: int = 16,
        hidden: int = 128,
        w_scale: float = 0.05,
    ):
        device = torch.device(device)
        C, H, Filt = int(channels), int(hidden), 4
        in_dim = C * Filt
        return dict(
            W1=torch.randn(B, H, in_dim, device=device) * w_scale,
            b1=torch.randn(B, H, device=device) * 0.0,
            W2=torch.randn(B, C, H, device=device) * w_scale,
            b2=torch.randn(B, C, device=device) * 0.0,
        )

    def seed(
        self,
        B: int = 1,
        H: int = 64,
        W: int = 64,
        mode: str = "noise",
        p: float = 0.1,
        generator=None,
    ) -> torch.Tensor:
        if generator is not None:
            return generator(
                B=B, H=H, W=W, device=self.device, mode=mode, p=p, C=self.C
            ).to(self.device)
        if mode == "zeros":
            return torch.zeros(B, self.C, H, W, device=self.device)
        if mode == "noise":
            return (torch.rand(B, self.C, H, W, device=self.device) - 0.5) * p
        if mode == "single":
            x = torch.zeros(B, self.C, H, W, device=self.device)
            x[:, :, H // 2, W // 2] = 1.0
            return x
        raise ValueError(f"unknown mode: {mode}")

    def forward(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        B, C, H, W = x.shape
        if C != self.C:
            raise ValueError(f"Expected C={self.C}, got {C}")

        y = depthwise_conv2d(x, self.filters)  # (B,C*F,H,W)
        y = y.permute(0, 2, 3, 1).reshape(B, H * W, self.in_dim)

        W1, b1 = params["W1"].to(x.device), params["b1"].to(x.device)
        W2, b2 = params["W2"].to(x.device), params["b2"].to(x.device)

        h = torch.einsum("bni,bhi->bnh", y, W1) + b1[:, None, :]
        h = F.relu(h)
        dx = torch.einsum("bnh,bch->bnc", h, W2) + b2[:, None, :]
        dx = dx.reshape(B, H, W, C).permute(0, 3, 1, 2)

        if self.fire_rate < 1.0:
            m = (torch.rand(B, 1, H, W, device=x.device) < self.fire_rate).float()
            dx = dx * m

        return x + self.dt * dx

    def to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(-2)
        return x[:, :3] if x.shape[1] >= 3 else super().to_rgb(x)


# -----------------------------
# Multi-channel Lenia (MCLenia)
# -----------------------------


class MCLenia2D(DynamicalSystem):
    """
    Multi-channel Lenia (repo's MCLenia) in functional form.

    State:  x  (B,C,H,W) in [0,1]
    Params (batched dict):
      mu       (B,C,C)
      sigma    (B,C,C)
      beta     (B,C,C,R)
      mu_k     (B,C,C,R)
      sigma_k  (B,C,C,R)
      weights  (B,C,C)   normalized so sum_i w_ij = 1
      (cached) fft_kernel (B,C,C,H,W) complex
    """

    def __init__(
        self,
        channels: int = 3,
        k_size: int = 25,
        dt: float = 0.1,
        rings: int = 3,
        device: str | torch.device = "cpu",
    ):
        super().__init__(device=device)
        self.C = int(channels)
        self.k_size = int(k_size)
        self.dt = float(dt)
        self.R = int(rings)

        if self.k_size <= 0:
            raise ValueError("k_size must be positive")
        if self.k_size % 2 == 0:
            self.k_size += 1  # repo: force odd

        # radius grid in [-1,1] (k,k)
        xy = torch.linspace(-1.0, 1.0, self.k_size, device=self.device)
        yy, xx = torch.meshgrid(xy, xy, indexing="ij")
        r = torch.sqrt(xx * xx + yy * yy)  # (k,k)
        self.register_buffer("r", r, persistent=False)

    # ----- parameter generation (repo distribution) -----

    @staticmethod
    def sample_params(
        B: int,
        device: str | torch.device = "cpu",
        *,
        channels: int = 3,
        k_size: int = 25,
        rings: int = 3,
        mode: str = "default",  # "default" | "random"
        H: int = 256,  # <-- NEW (defaults to your common case)
        W: int = 256,  # <-- NEW
        precompute_fft: bool = True,  # <-- NEW
    ):
        """
        Repo-like random params + (optionally) precompute fft_kernel for (H,W).
        Returned dict is ready to use directly in forward/rollout.
        """
        dev = torch.device(device)
        C = int(channels)
        R = int(rings)
        k = int(k_size)
        if k % 2 == 0:
            k += 1

        if mode not in ("default", "random"):
            raise ValueError("mode must be 'default' or 'random'")

        # --- sample like LeniaParams.default_gen/random_gen ---
        if mode == "default":
            mu = 0.7 * torch.rand((B, C, C), device=dev)
            sigma = (
                mu
                / (math.sqrt(2 * math.log(2)))
                * 0.8
                * torch.rand((B, C, C), device=dev)
                + 1e-4
            )
            mu_k = 0.5 + 0.2 * torch.randn((B, C, C, R), device=dev)
        else:
            mu = torch.rand((B, C, C), device=dev)
            sigma = torch.rand((B, C, C), device=dev) + 1e-4
            mu_k = torch.rand((B, C, C, R), device=dev)

        beta = torch.rand((B, C, C, R), device=dev)
        sigma_k = 0.05 * (
            1
            + torch.clamp(0.3 * torch.randn((B, C, C, R), device=dev), min=-0.9)
            + 1e-4
        )

        w = torch.rand((B, C, C), device=dev)
        w = (
            w * (1.0 - 0.8 * torch.eye(C, device=dev))[None, :, :]
        )  # suppress diagonal a bit

        # sanitize
        mu = mu.clamp(0.0, 2.0)
        sigma = sigma.clamp_min(1e-6)
        mu_k = mu_k.clamp(0.0, 2.0)
        sigma_k = sigma_k.clamp_min(1e-6)
        w = w.clamp_min(0.0)

        # normalize weights so sum_i w_ij = 1
        N = w.sum(dim=1, keepdim=True)
        w = torch.where(N > 1e-6, w / N, torch.zeros_like(w))

        params = dict(
            k_size=k,
            mu=mu,
            sigma=sigma,
            beta=beta,
            mu_k=mu_k,
            sigma_k=sigma_k,
            weights=w,
        )

        # --- NEW: precompute kernel + fft_kernel for the requested grid ---
        if precompute_fft:
            if H < k or W < k:
                raise ValueError(f"(H,W)=({H},{W}) must be >= k_size={k}")

            # radius grid r in [-1,1] (k,k)
            xy = torch.linspace(-1.0, 1.0, k, device=dev)
            yy, xx = torch.meshgrid(xy, xy, indexing="ij")
            r = torch.sqrt(xx * xx + yy * yy)[None, None, None, None]  # (1,1,1,1,k,k)

            # kernel rings -> kernel (B,C,C,k,k)
            mu_k_e = mu_k[..., None, None]  # (B,C,C,R,1,1)
            sg_k_e = sigma_k[..., None, None] + 1e-8  # (B,C,C,R,1,1)
            beta_e = beta[..., None, None]  # (B,C,C,R,1,1)
            rings_val = torch.exp(-0.5 * ((r - mu_k_e) / sg_k_e) ** 2)  # (B,C,C,R,k,k)
            K = (beta_e * rings_val).sum(dim=3)  # (B,C,C,k,k)

            # normalize integral(K)=1 per (B,C,C)
            s = K.sum(dim=(-1, -2), keepdim=True)
            s = torch.where(s < 1e-6, torch.ones_like(s), s)
            K = K / s

            # pad + center + FFT -> (B,C,C,H,W) complex
            Kp = F.pad(K, (0, W - k, 0, H - k))  # (B,C,C,H,W)
            Kp = torch.roll(Kp, shifts=(-k // 2, -k // 2), dims=(-2, -1))
            params["fft_kernel"] = torch.fft.fft2(Kp)

        return params

    @staticmethod
    def iter_params(*args, **kwargs):
        raise NotImplementedError("Continuous parameter space; use sample_params().")

    # ----- init state -----

    def seed(
        self,
        B: int = 1,
        H: int = 128,
        W: int = 128,
        mode: str = "fractal",  # "fractal" | "noise" | "zeros" | "circle"
        p: float = 0.25,
        radius: float | None = None,
        generator=None,
    ) -> torch.Tensor:
        if generator is not None:
            return generator(
                B=B,
                H=H,
                W=W,
                C=self.C,
                device=self.device,
                mode=mode,
                p=p,
                radius=radius,
            ).to(self.device)

        if mode == "zeros":
            return torch.zeros(B, self.C, H, W, device=self.device)
        if mode == "noise":
            return (torch.rand(B, self.C, H, W, device=self.device) * p).clamp(0.0, 1.0)
        if mode == "fractal":
            return fractal_noise_2d(
                B,
                self.C,
                H,
                W,
                device=self.device,
                black_prop=0.25,
                octaves=4,
                persistence=0.4,
            )
        if mode == "circle":
            if radius is None:
                radius = self.k_size * 3
            x = fractal_noise_2d(B, self.C, H, W, device=self.device, black_prop=0.25)
            yy, xx = torch.meshgrid(
                torch.linspace(-H // 2, H // 2, H, device=self.device),
                torch.linspace(-W // 2, W // 2, W, device=self.device),
                indexing="ij",
            )
            R = torch.sqrt(xx * xx + yy * yy)
            mask = (R < float(radius))[None, None]
            return torch.where(mask, x, torch.zeros_like(x))
        raise ValueError(f"unknown seed mode: {mode}")

    # ----- shared core methods (used by child too) -----

    @staticmethod
    def _norm_weights(w: torch.Tensor) -> torch.Tensor:
        # enforce sum_i w_ij = 1 (sum over input channel dim=1)
        N = w.sum(dim=1, keepdim=True)
        return torch.where(N > 1e-6, w / N, torch.zeros_like(w))

    def _compute_kernel(self, params: dict) -> torch.Tensor:
        """
        Kernel K: (B,C,C,k,k), normalized over (k,k) per (B,C,C).
        """
        mu_k = params["mu_k"].to(self.device, torch.float32)  # (B,C,C,R)
        sigma_k = params["sigma_k"].to(self.device, torch.float32)  # (B,C,C,R)
        beta = params["beta"].to(self.device, torch.float32)  # (B,C,C,R)

        B, C, _, R = mu_k.shape
        k = self.k_size

        r = self.r.to(self.device, torch.float32)[
            None, None, None, None
        ]  # (1,1,1,1,k,k)
        r = r.expand(B, C, C, R, k, k)  # (B,C,C,R,k,k)

        mu_k = mu_k[..., None, None]  # (B,C,C,R,1,1)
        sigma_k = sigma_k[..., None, None].clamp_min(1e-8)
        beta = beta[..., None, None]  # (B,C,C,R,1,1)

        rings = torch.exp(-0.5 * ((r - mu_k) / sigma_k) ** 2)  # (B,C,C,R,k,k)
        K = (beta * rings).sum(dim=3)  # (B,C,C,k,k)

        s = K.sum(dim=(-1, -2), keepdim=True)
        s = torch.where(s < 1e-6, torch.ones_like(s), s)
        return K / s

    def _kernel_to_fft(self, K: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        K: (B,C,C,k,k) -> fft_kernel: (B,C,C,H,W) complex
        """
        k = self.k_size
        if H < k or W < k:
            raise ValueError(f"grid too small: (H,W)=({H},{W}) < k={k}")

        Kp = F.pad(K, (0, W - k, 0, H - k))  # (B,C,C,H,W)
        Kp = torch.roll(Kp, shifts=(-k // 2, -k // 2), dims=(-2, -1))
        return torch.fft.fft2(Kp)

    def prepare_params(self, params: dict, H: int, W: int) -> dict:
        """
        Precompute fft_kernel into params (in-place). Call once before rollout for speed.
        """
        # ensure consistent k_size if provided
        k = params.get("k_size", self.k_size)
        if isinstance(k, torch.Tensor):
            k = int(k.item())
        if int(k) != int(self.k_size):
            raise ValueError(
                f"params k_size={k} does not match model k_size={self.k_size}"
            )

        K = self._compute_kernel(params)  # (B,C,C,k,k)
        params["fft_kernel"] = self._kernel_to_fft(K, H, W)  # complex
        return params

    def fftconv(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Convolution via cached fft_kernel.
        Returns U: (B,C,C,H,W)
        """
        x = x.to(self.device, torch.float32)
        B, C, H, W = x.shape
        if C != self.C:
            raise ValueError(f"Expected C={self.C}, got {C}")

        fft_k = params.get("fft_kernel", None)
        if (fft_k is None) or (fft_k.shape[-2:] != (H, W)):
            self.prepare_params(params, H, W)
            fft_k = params["fft_kernel"]

        fft_k = fft_k.to(self.device)
        X = torch.fft.fft2(x)  # (B,C,H,W)
        U = torch.fft.ifft2(X[:, :, None] * fft_k)  # (B,C,C,H,W)
        return torch.real(U)

    def growth(self, U: torch.Tensor, params: dict) -> torch.Tensor:
        """
        U: (B,C,C,H,W) -> growth: (B,C,C,H,W) in ~[-1,1]
        """
        mu = params["mu"].to(self.device, torch.float32)[..., None, None]  # (B,C,C,1,1)
        sigma = (
            params["sigma"]
            .to(self.device, torch.float32)[..., None, None]
            .clamp_min(1e-8)
        )
        return 2.0 * torch.exp(-0.5 * ((U - mu) / sigma) ** 2) - 1.0

    # ----- dynamics -----

    def forward(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Standard multi-channel Lenia step:
          x <- clamp(x + dt * sum_i w_ij * G(U_ij), 0, 1)
        """
        x = x.to(self.device, torch.float32)

        w = params["weights"].to(self.device, torch.float32)
        w = self._norm_weights(w)

        U = self.fftconv(x, params)  # (B,C,C,H,W)
        dx = (self.growth(U, params) * w[..., None, None]).sum(
            dim=1
        )  # sum over input channel -> (B,C,H,W)
        return (x + self.dt * dx).clamp(0.0, 1.0)

    def mass(self, x: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) -> (B,C) mean mass per channel."""
        return x.mean(dim=(-1, -2))

    def to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        # (B,C,H,W) -> (B,3,H,W)
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        if x.shape[1] == 2:
            z = torch.zeros_like(x[:, :1])
            return torch.cat([x, z], dim=1)
        return x[:, :3]


# ---------------------------------------
# Mass-conserving "MaceLenia" (Diffusion)
# ---------------------------------------


class DiffusionLenia2D(MCLenia2D):
    """
    Mass-conserving Lenia-like model (repo's DiffusionLenia).

    Step:
      Aff = exp(temp * sum_i w_ij * G(U_ij))
      Z   = sum_{n in N3x3} Aff_n
      x'  = Aff * sum_{n in N3x3} (x_n / Z_n)

    This conserves total mass per channel (up to numerical error).
    """

    def __init__(
        self,
        channels: int = 3,
        k_size: int = 25,
        dt: float = 0.1,  # kept for API consistency; not used in diffusion step (repo behavior)
        rings: int = 3,
        temp: float = 1.0,
        device: str | torch.device = "cpu",
    ):
        super().__init__(
            channels=channels, k_size=k_size, dt=dt, rings=rings, device=device
        )
        self._temp = float(temp)

        # depthwise 3x3 ones kernel for local sums
        k = torch.ones(self.C, 1, 3, 3, device=self.device, dtype=torch.float32)
        self.register_buffer("_ones3", k, persistent=False)

    @property
    def temp(self) -> float:
        return self._temp

    @temp.setter
    def temp(self, value: float):
        self._temp = float(value)

    def compute_affinity(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Returns Aff: (B,C,H,W), strictly positive.
        """
        x = x.to(self.device, torch.float32)

        w = params["weights"].to(self.device, torch.float32)
        w = self._norm_weights(w)

        U = self.fftconv(x, params)  # (B,C,C,H,W)
        pre = (self.growth(U, params) * w[..., None, None]).sum(dim=1)  # (B,C,H,W)

        # avoid exp overflow
        pre = (self._temp * pre).clamp(-50.0, 50.0)
        return torch.exp(pre)

    def forward(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """
        Mass-conserving diffusion step (no clamp).
        """
        x = x.to(self.device, torch.float32)
        Aff = self.compute_affinity(x, params)  # (B,C,H,W)

        # Z = local sum of affinities (3x3, circular), per-channel
        Z = F.conv2d(
            F.pad(Aff, (1, 1, 1, 1), mode="circular"), self._ones3, groups=self.C
        )  # (B,C,H,W)
        Z = Z.clamp_min(1e-8)

        portions = x / Z
        neigh = F.conv2d(
            F.pad(portions, (1, 1, 1, 1), mode="circular"), self._ones3, groups=self.C
        )  # (B,C,H,W)

        return Aff * neigh


class LeniaJAX2D(MCLenia2D):
    """
    JAX-style Lenia (MultiLeniaJAX) on top of MCLenia2D.

    Differences vs MCLenia2D:
      - supports params["func_k"] in {"gauss","exp","quad4"}
      - exp/quad4 use Bert-style ring segmentation driven by beta
      - optional voronoi polygon seeding (mode="voronoi")
      - FFT kernel placement matches the JAX code (center placement then roll by -H//2,-W//2)
    """

    # -------------------------
    # params
    # -------------------------

    @staticmethod
    def sample_params(
        B: int,
        device: str | torch.device = "cpu",
        *,
        # main hyperparams
        channels: int = 1,
        k_size: int = 27,
        rings: int = 1,                 # for gauss: #cores, for exp/quad4: #beta rings
        func_k: str = "gauss",           # "gauss" | "exp" | "quad4"

        # optional explicit params (floats/lists/arrays/tensors ok)
        mu=None,
        sigma=None,
        beta=None,
        mu_k=None,
        sigma_k=None,
        weights=None,

        # random ranges (used when not explicitly provided)
        mu_range=(0.05, 0.25),
        sigma_range=(0.005, 0.03),
        mu_k_range=(0.2, 0.8),
        sigma_k_range=(0.05, 0.25),

        # precompute fft kernel (like your JAX class)
        H: int = 256,
        W: int = 256,
        precompute_fft: bool = True,
    ) -> Dict[str, torch.Tensor]:
        dev = torch.device(device)
        C = int(channels)
        k = int(k_size)
        if k % 2 == 0:
            k += 1
        func_k = str(func_k).lower()
        if func_k not in ("gauss", "exp", "quad4"):
            raise ValueError("func_k must be one of: 'gauss', 'exp', 'quad4'")

        def to_torch(v, dtype=torch.float32):
            if v is None:
                return None
            # handles numpy, jax arrays, lists, scalars
            return torch.tensor(np.asarray(v), device=dev, dtype=dtype)

        def expand_batch(t: torch.Tensor, target_ndim: int) -> torch.Tensor:
            # if missing batch dim -> add; if batch=1 -> expand to B
            if t.ndim == target_ndim - 1:
                t = t.unsqueeze(0)
            if t.shape[0] == 1 and B > 1:
                t = t.expand(B, *t.shape[1:]).contiguous()
            if t.shape[0] != B:
                raise ValueError(f"Expected batch {B}, got {t.shape[0]}")
            return t

        # --- growth params mu/sigma: (B,C,C)
        mu_t = to_torch(mu)
        if mu_t is None:
            mu_t = mu_range[0] + (mu_range[1] - mu_range[0]) * torch.rand(B, C, C, device=dev)
        else:
            mu_t = expand_batch(mu_t.float(), 3)
        mu_t = mu_t.clamp(0.0, 2.0)

        sig_t = to_torch(sigma)
        if sig_t is None:
            sig_t = sigma_range[0] + (sigma_range[1] - sigma_range[0]) * torch.rand(B, C, C, device=dev)
        else:
            sig_t = expand_batch(sig_t.float(), 3)
        sig_t = sig_t.clamp_min(1e-6)

        # --- beta: (B,C,C,R)
        beta_t = to_torch(beta)
        if beta_t is None:
            R = int(rings)
            beta_t = torch.rand(B, C, C, R, device=dev)
        else:
            # accept shape (R,) or (1,1,1,R) etc.
            if beta_t.ndim == 1:
                beta_t = beta_t.view(1, 1, 1, -1).expand(B, C, C, -1).contiguous()
            elif beta_t.ndim == 4:
                beta_t = expand_batch(beta_t.float(), 4)
                if beta_t.shape[1:] != (C, C, beta_t.shape[3]):
                    # if passed as (B,1,1,R) but channels>1: expand across pairs
                    if beta_t.shape[1] == 1 and beta_t.shape[2] == 1:
                        beta_t = beta_t.expand(B, C, C, beta_t.shape[3]).contiguous()
            else:
                raise ValueError("beta must be 1D (R,) or 4D (B,C,C,R)")
        beta_t = beta_t.clamp_min(0.0)

        # --- mu_k/sigma_k: only used for func_k="gauss" (but we keep them always for compatibility)
        R = beta_t.shape[3]
        mu_k_t = to_torch(mu_k)
        if mu_k_t is None:
            mu_k_t = mu_k_range[0] + (mu_k_range[1] - mu_k_range[0]) * torch.rand(B, C, C, R, device=dev)
        else:
            if mu_k_t.ndim == 1:
                mu_k_t = mu_k_t.view(1, 1, 1, -1).expand(B, C, C, -1).contiguous()
            else:
                mu_k_t = expand_batch(mu_k_t.float(), 4)
                if mu_k_t.shape[1] == 1 and mu_k_t.shape[2] == 1 and C > 1:
                    mu_k_t = mu_k_t.expand(B, C, C, mu_k_t.shape[3]).contiguous()
        mu_k_t = mu_k_t.clamp(0.0, 2.0)

        sig_k_t = to_torch(sigma_k)
        if sig_k_t is None:
            sig_k_t = sigma_k_range[0] + (sigma_k_range[1] - sigma_k_range[0]) * torch.rand(B, C, C, R, device=dev)
        else:
            if sig_k_t.ndim == 1:
                sig_k_t = sig_k_t.view(1, 1, 1, -1).expand(B, C, C, -1).contiguous()
            else:
                sig_k_t = expand_batch(sig_k_t.float(), 4)
                if sig_k_t.shape[1] == 1 and sig_k_t.shape[2] == 1 and C > 1:
                    sig_k_t = sig_k_t.expand(B, C, C, sig_k_t.shape[3]).contiguous()
        sig_k_t = sig_k_t.clamp_min(1e-6)

        # --- weights: (B,C,C)
        w_t = to_torch(weights)
        if w_t is None:
            if C == 1:
                w_t = torch.ones(B, 1, 1, device=dev)
            else:
                w_t = torch.rand(B, C, C, device=dev)
        else:
            w_t = expand_batch(w_t.float(), 3)
            if w_t.shape[1] == 1 and w_t.shape[2] == 1 and C > 1:
                w_t = w_t.expand(B, C, C).contiguous()
        # normalize sum_i w_ij = 1
        N = w_t.sum(dim=1, keepdim=True)
        w_t = torch.where(N > 1e-6, w_t / N, torch.zeros_like(w_t))

        params = dict(
            k_size=k,
            mu=mu_t,
            sigma=sig_t,
            beta=beta_t,
            mu_k=mu_k_t,
            sigma_k=sig_k_t,
            weights=w_t,
            func_k=func_k,
        )

        if precompute_fft:
            if H < k or W < k:
                raise ValueError(f"(H,W)=({H},{W}) must be >= k_size={k}")

            # build radius grid like JAX (xy in [-1,1])
            xy = torch.linspace(-1.0, 1.0, k, device=dev)
            X, Y = torch.meshgrid(xy, xy, indexing="xy")
            r = torch.sqrt(X * X + Y * Y)  # (k,k)

            # compute kernel K: (B,C,C,k,k)
            K = LeniaJAX2D._compute_kernel_static(r, params)

            # JAX-style kernel_to_fft: center-insert then roll by -H//2,-W//2
            Bp, Cp, Cp2, _, _ = K.shape
            padded = torch.zeros(Bp, Cp, Cp2, H, W, device=dev, dtype=torch.float32)
            sh = H // 2 - k // 2
            sw = W // 2 - k // 2
            padded[:, :, :, sh : sh + k, sw : sw + k] = K
            padded = torch.roll(padded, shifts=(-H // 2, -W // 2), dims=(-2, -1))
            params["fft_kernel"] = torch.fft.fft2(padded)

        return params

    @staticmethod
    def iter_params(*args, **kwargs):
        raise NotImplementedError("Continuous parameter space; use sample_params().")

    # -------------------------
    # kernel (override)
    # -------------------------

    @staticmethod
    def _compute_kernel_static(r: torch.Tensor, params: dict) -> torch.Tensor:
        """
        r: (k,k) float
        returns K: (B,C,C,k,k) normalized
        """
        func = str(params.get("func_k", "gauss")).lower()
        beta = params["beta"].float()  # (B,C,C,R)
        B, C, _, R = beta.shape
        k = r.shape[0]
        dev = beta.device

        if func == "gauss":
            mu_k = params["mu_k"].float()       # (B,C,C,R)
            sg_k = params["sigma_k"].float()    # (B,C,C,R)
            rr = r.to(dev)[None, None, None, None]                    # (1,1,1,1,k,k)
            rr = rr.expand(B, C, C, R, k, k)
            mu_k = mu_k[..., None, None]
            sg_k = sg_k[..., None, None].clamp_min(1e-8)
            beta_e = beta[..., None, None]
            rings = torch.exp(-0.5 * ((rr - mu_k) / sg_k) ** 2)
            K = (beta_e * rings).sum(dim=3)  # (B,C,C,k,k)

        elif func in ("exp", "quad4"):
            # JAX: r = where(r>1,0,r)
            rr = torch.where(r.to(dev) > 1.0, torch.zeros_like(r.to(dev)), r.to(dev))  # (k,k)
            b = R

            if b > 1:
                Br = b * rr
                seg = torch.floor(Br).long().clamp_(0, b - 1)  # (k,k)
                rf = Br - seg.float()                          # (k,k) in [0,1)
                beta_map = beta[..., seg]                      # (B,C,C,k,k)
            else:
                rf = rr
                beta_map = beta[..., 0].unsqueeze(-1).unsqueeze(-1)  # (B,C,C,1,1)

            if func == "exp":
                denom = rf * (1.0 - rf)
                base = torch.exp(4.0 - 1.0 / (denom + 1e-8))          # (k,k)
            else:  # quad4
                base = (4.0 * rf * (1.0 - rf)).clamp_min(0.0) ** 4    # (k,k)

            K = beta_map * base  # broadcast to (B,C,C,k,k)

        else:
            raise ValueError(f"unknown func_k: {func}")

        # normalize integral(K)=1 per (B,C,C)
        s = K.sum(dim=(-1, -2), keepdim=True)
        s = torch.where(s < 1e-6, torch.ones_like(s), s)
        return K / s

    def _compute_kernel(self, params: dict) -> torch.Tensor:
        # use the same logic but with this instance's radius grid self.r
        r = self.r.to(self.device, torch.float32)
        return self._compute_kernel_static(r, params)

    def _kernel_to_fft(self, K: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Override to match the JAX kernel_to_fft (center insert then roll by -H//2,-W//2).
        """
        B, C, C2, k, _ = K.shape
        if H < k or W < k:
            raise ValueError(f"(H,W)=({H},{W}) must be >= k_size={k}")
        padded = torch.zeros(B, C, C2, H, W, device=K.device, dtype=K.dtype)
        sh = H // 2 - k // 2
        sw = W // 2 - k // 2
        padded[:, :, :, sh : sh + k, sw : sw + k] = K
        padded = torch.roll(padded, shifts=(-H // 2, -W // 2), dims=(-2, -1))
        return torch.fft.fft2(padded)

    # -------------------------
    # init state (add voronoi)
    # -------------------------

    def seed(
        self,
        B: int = 1,
        H: int = 128,
        W: int = 128,
        mode: str = "voronoi",   # "voronoi" | (fallback to parent modes: "fractal","noise","zeros","circle")
        polygon_size: int = 60,
        init_polygon_index: int = 0,
        seeds: Optional[list[int]] = None,
        p: float = 0.25,         # used for "noise"/parent modes
        radius: float | None = None,
        generator=None,
    ) -> torch.Tensor:
        if generator is not None:
            return generator(
                B=B, H=H, W=W, C=self.C, device=self.device, mode=mode,
                polygon_size=polygon_size, init_polygon_index=init_polygon_index, seeds=seeds,
                p=p, radius=radius
            ).to(self.device)

        mode = str(mode).lower()
        if mode != "voronoi":
            # reuse MCLenia2D's modes ("zeros","noise","fractal","circle")
            return super().seed(B=B, H=H, W=W, mode=mode, p=p, radius=radius, generator=None)

        # --- voronoi init (matches your JAX logic) ---
        import os, pickle

        path = f"utils/polygons{H}.pickle"
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path}. Generate polygons first (your voronoi script) or choose another seed mode."
            )

        with open(path, "rb") as f:
            data = pickle.load(f)
        if polygon_size not in data:
            raise ValueError(f"polygon_size={polygon_size} not in {path}. Available keys: {list(data.keys())[:10]}...")

        masks = data[polygon_size]
        if len(masks) == 0:
            raise ValueError(f"No masks for polygon_size={polygon_size} in {path}")

        # helper: embed a small mask into (H,W) centered (like load_pattern)
        def embed_center(mask_np: np.ndarray) -> np.ndarray:
            mask_np = np.asarray(mask_np, dtype=np.float32)
            hh, ww = mask_np.shape
            out = np.zeros((H, W), dtype=np.float32)
            x1 = H // 2 - hh // 2
            y1 = W // 2 - ww // 2
            out[x1 : x1 + hh, y1 : y1 + ww] = mask_np
            return out

        x = torch.zeros(B, self.C, H, W, device=self.device, dtype=torch.float32)

        if seeds is not None and len(seeds) != B:
            raise ValueError("If provided, seeds must have length B")

        for i in range(B):
            mask = masks[(init_polygon_index + i) % len(masks)]
            m = torch.tensor(embed_center(mask), device=self.device, dtype=torch.float32)[None, None]  # (1,1,H,W)

            if seeds is None:
                noise = torch.rand(1, self.C, H, W, device=self.device)
            else:
                g = torch.Generator(device=self.device)
                g.manual_seed(int(seeds[i]))
                noise = torch.rand(1, self.C, H, W, device=self.device, generator=g)

            x[i : i + 1] = noise * m

        return x



# -----------------------------
# Gray-Scott reaction-diffusion (bonus 2D)
# -----------------------------


class GrayScott2D(DynamicalSystem):
    """
    State: (B,2,H,W) with channels (u,v)
    Params: Du,Dv,F,k as (B,)
    """

    def __init__(self, dt: float = 1.0, device: str | torch.device = "cpu"):
        super().__init__(device=device)
        self.dt = float(dt)
        lap = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=self.device
        ).float()
        self.register_buffer("lap", lap.view(1, 3, 3), persistent=False)

    @staticmethod
    def sample_params(
        B: int,
        device: str | torch.device = "cpu",
        Du=(0.12, 0.2),
        Dv=(0.05, 0.12),
        F_=(0.01, 0.06),
        k_=(0.03, 0.07),
    ):
        device = torch.device(device)
        return dict(
            Du=Du[0] + (Du[1] - Du[0]) * torch.rand(B, device=device),
            Dv=Dv[0] + (Dv[1] - Dv[0]) * torch.rand(B, device=device),
            F=F_[0] + (F_[1] - F_[0]) * torch.rand(B, device=device),
            k=k_[0] + (k_[1] - k_[0]) * torch.rand(B, device=device),
        )

    def seed(
        self,
        B: int = 1,
        H: int = 128,
        W: int = 128,
        noise: float = 0.02,
        generator=None,
    ) -> torch.Tensor:
        if generator is not None:
            return generator(B=B, H=H, W=W, device=self.device, noise=noise).to(
                self.device
            )

        u = torch.ones(B, 1, H, W, device=self.device)
        v = torch.zeros(B, 1, H, W, device=self.device)
        r = min(H, W) // 10
        v[:, :, H // 2 - r : H // 2 + r, W // 2 - r : W // 2 + r] = 1.0
        u[:, :, H // 2 - r : H // 2 + r, W // 2 - r : W // 2 + r] = 0.0
        if noise > 0:
            u = (u + noise * torch.randn_like(u)).clamp(0, 1)
            v = (v + noise * torch.randn_like(v)).clamp(0, 1)
        return torch.cat([u, v], dim=1)

    def forward(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        B, C, H, W = x.shape
        if C != 2:
            raise ValueError("GrayScott2D expects C=2 (u,v)")
        u, v = x[:, :1], x[:, 1:2]
        lap_uv = depthwise_conv2d(x, self.lap)  # (B,2,H,W)
        lap_u, lap_v = lap_uv[:, :1], lap_uv[:, 1:2]

        Du = _as_tensor(params["Du"], x.device).view(B, 1, 1, 1)
        Dv = _as_tensor(params["Dv"], x.device).view(B, 1, 1, 1)
        Ff = _as_tensor(params["F"], x.device).view(B, 1, 1, 1)
        kk = _as_tensor(params["k"], x.device).view(B, 1, 1, 1)

        uvv = u * v * v
        du = Du * lap_u - uvv + Ff * (1.0 - u)
        dv = Dv * lap_v + uvv - (Ff + kk) * v
        return torch.cat([u + self.dt * du, v + self.dt * dv], dim=1).clamp(0.0, 1.0)

    def to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(-2)
        g = x[:, 1:2] if x.shape[1] >= 2 else x.mean(dim=1, keepdim=True)
        return g.repeat(1, 3, 1, 1)


# -----------------------------
# Coupled logistic map lattice (bonus 1D)
# -----------------------------


class CoupledLogistic1D(DynamicalSystem):
    """
    f(x)=r*x*(1-x)
    x_{t+1}=(1-eps)*f(x_i) + eps/2*(f(x_{i-1}) + f(x_{i+1}))
    """

    @staticmethod
    def sample_params(
        B: int, device: str | torch.device = "cpu", r=(3.5, 4.0), eps=(0.0, 0.5)
    ):
        device = torch.device(device)
        return dict(
            r=r[0] + (r[1] - r[0]) * torch.rand(B, device=device),
            eps=eps[0] + (eps[1] - eps[0]) * torch.rand(B, device=device),
        )

    def seed(
        self, B: int = 1, W: int = 256, mode: str = "noise", generator=None
    ) -> torch.Tensor:
        if generator is not None:
            return generator(B=B, W=W, device=self.device, mode=mode).to(self.device)
        if mode == "noise":
            return torch.rand(B, 1, W, device=self.device).clamp(0, 1)
        raise ValueError(f"unknown mode: {mode}")

    def forward(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        B, C, W = x.shape
        if C != 1:
            raise ValueError("CoupledLogistic1D expects C=1")

        r = _as_tensor(params["r"], x.device).view(B, 1, 1)
        eps = _as_tensor(params["eps"], x.device).view(B, 1, 1)

        f = r * x * (1.0 - x)
        fl = torch.roll(f, 1, dims=-1)
        fr = torch.roll(f, -1, dims=-1)
        return ((1.0 - eps) * f + 0.5 * eps * (fl + fr)).clamp(0.0, 1.0)

    def to_integer(self, x: torch.Tensor, num_bins: int = 2) -> torch.Tensor:
        """Quantize continuous [0,1] values to num_bins integer levels."""
        x_f = x.float().clamp(0.0, 1.0)
        return (x_f * num_bins).floor().long().clamp(0, num_bins - 1)

# -----------------------------
# Langton-style implicit-rule CA (1D / 2D)  — UPDATED
#   * lambda now follows Langton's standard:
#       lambda = fraction of NON-QUIESCENT outputs
#       (i.e. fraction of neighborhoods that map to state != 0)
#   * Added methods to compute actual lambda / quiescent fraction by exhaustive evaluation
# -----------------------------


class LangtonCABase(DynamicalSystem):
    """
    Base for Langton-style CAs with an *implicit* random rule table.

    For each neighborhood configuration, compute a deterministic keyed hash
    (keyed by params['seed']) -> pseudo-random scalar u in [0, 1).

    Langton semantics (STANDARD):
      - lambda = fraction of NON-QUIESCENT transitions (output != 0)
      - equivalently: P(output==0) = 1 - lambda

    So we implement:
      - if u < (1 - lambda): return 0 (quiescent)
      - else: return one of {1..K-1} uniformly (via binning)

    Notes:
      - Everything is deterministic given (seed, neighborhood).
      - Uses safe 32-bit modular arithmetic (avoids signed int overflow issues).
    """

    # Spatial dimensionality, set by subclass (1 for 1D, 2 for 2D)
    _spatial_ndim: int = 0
    discrete_state = True

    # 31-bit uniform domain for u (so MOD=2^31)
    _MOD31: int = 1 << 31
    _MASK32: int = 0xFFFFFFFF
    _MASK31: int = 0x7FFFFFFF

    # MurmurHash3 32-bit finalizer constants
    _FMIX_MUL1: int = 0x7FEB352D
    _FMIX_MUL2: int = 0x846CA68B

    # Seed mixing constants
    _SEED_MUL: int = 0x9E3779B1
    _ACC_OFFSET: int = 0x85EBCA6B

    def __init__(
        self,
        *,
        num_states: int,
        default_lambda: float = 0.5,   # Langton lambda (non-quiescent fraction)
        default_seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        super().__init__(device=device)
        K = int(num_states)
        if K < 1:
            raise ValueError("num_states must be >= 1")
        self.num_states = K

        self.register_buffer(
            "_default_lambda",
            torch.tensor(float(default_lambda), device=self.device, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_default_seed",
            torch.tensor(int(default_seed), device=self.device, dtype=torch.int64),
            persistent=False,
        )

        # Color palette (K,3) in [0,1]
        palette = self._make_palette(K, device=self.device)
        self.register_buffer("_palette", palette, persistent=False)

        # Subclasses should set these:
        #   self.neighborhood_len: int
        #   self._coeff_flat: (L,) int32 buffer

    # -------------------------
    # Color palette (nice RGB)
    # -------------------------

    @staticmethod
    def _hsv_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
        """
        hsv: (N,3) in [0,1]
        returns rgb: (N,3) in [0,1]
        """
        if hsv.ndim != 2 or hsv.shape[1] != 3:
            raise ValueError("hsv must have shape (N,3)")
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        h6 = h * 6.0
        i = torch.floor(h6).to(torch.int64)
        f = h6 - i.to(h6.dtype)

        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)

        i = (i % 6).to(torch.int64)

        rgb0 = torch.stack([v, t, p], dim=1)
        rgb1 = torch.stack([q, v, p], dim=1)
        rgb2 = torch.stack([p, v, t], dim=1)
        rgb3 = torch.stack([p, q, v], dim=1)
        rgb4 = torch.stack([t, p, v], dim=1)
        rgb5 = torch.stack([v, p, q], dim=1)

        cases = torch.stack([rgb0, rgb1, rgb2, rgb3, rgb4, rgb5], dim=1)  # (N,6,3)
        idx = i.view(-1, 1, 1).expand(-1, 1, 3)  # (N,1,3)
        return cases.gather(1, idx).squeeze(1)

    @classmethod
    def _make_palette(cls, K: int, device: torch.device) -> torch.Tensor:
        """
        (K,3) float palette in [0,1].
          state 0 -> black
          states 1..K-1 -> evenly spread hues using golden-ratio spacing
        """
        pal = torch.zeros((K, 3), device=device, dtype=torch.float32)
        if K <= 1:
            return pal
        if K == 2:
            pal[1] = torch.tensor([1.0, 1.0, 1.0], device=device)
            return pal

        i = torch.arange(1, K, device=device, dtype=torch.float32)
        hue = torch.frac(i * 0.6180339887498949)  # golden ratio conjugate
        sat = torch.full_like(hue, 0.85)
        val = torch.full_like(hue, 1.0)
        hsv = torch.stack([hue, sat, val], dim=1)
        pal[1:] = cls._hsv_to_rgb(hsv)
        return pal

    def to_float(self, x: torch.Tensor) -> torch.Tensor:
        """
        One-hot encode integer states to float multi-channel representation.

        State (B,1,*spatial) -> (B,K,*spatial)
        Rollout (B,T,1,*spatial) -> (B,T,K,*spatial)
        """
        K = self.num_states
        if K <= 1:
            return x.float()
        x_int = x.long().clamp(0, K - 1)

        ndim_state = self._spatial_ndim + 2  # B + C + spatial dims
        if x_int.ndim == ndim_state:
            # Single state: (B,1,*spatial) -> (B,K,*spatial)
            x_squeezed = x_int.squeeze(1)          # (B,*spatial)
            oh = F.one_hot(x_squeezed, K).float()  # (B,*spatial,K)
            return oh.movedim(-1, 1)                # (B,K,*spatial)
        elif x_int.ndim == ndim_state + 1:
            # Rollout: (B,T,1,*spatial) -> (B,T,K,*spatial)
            x_squeezed = x_int.squeeze(2)          # (B,T,*spatial)
            oh = F.one_hot(x_squeezed, K).float()  # (B,T,*spatial,K)
            return oh.movedim(-1, 2)                # (B,T,K,*spatial)
        return x.float()

    def to_integer(self, x: torch.Tensor, num_bins: int = 2) -> torch.Tensor:
        """Return integer states (already integer). Clamp to valid range."""
        if x.is_floating_point():
            return x.round().long().clamp(0, self.num_states - 1)
        return x.long().clamp(0, self.num_states - 1)

    def to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map integer states -> RGB.

        Accepts:
          - (B,1,W)   -> (B,3,1,W)
          - (B,1,H,W) -> (B,3,H,W)

        Output is float in [0,1].
        """
        # Normalize to (B,1,H,W)
        if x.ndim == 3:
            x = x.unsqueeze(-2)
        if x.ndim != 4:
            raise ValueError("to_rgb expects (B,1,W) or (B,1,H,W)")
        if x.shape[1] != 1:
            x = x[:, :1]

        idx = x.squeeze(1).to(dtype=torch.long)  # (B,H,W)
        pal = self._palette.to(device=x.device)
        rgb = F.embedding(idx, pal)  # (B,H,W,3)
        return rgb.permute(0, 3, 1, 2).contiguous()  # (B,3,H,W)

    # -------------------------
    # Parameter sampling
    # -------------------------

    @staticmethod
    def _splitmix64_next(x: int) -> tuple[int, int]:
        """Pure-python SplitMix64 step (returns (new_x, z))."""
        x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = x
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        z = z ^ (z >> 31)
        return x, z

    @classmethod
    def _make_coeff_int32(cls, L: int) -> list[int]:
        """
        Make L signed int32 coefficients (python ints in [-2^31, 2^31-1]).
        Deterministic and independent of the rule seed.
        """
        if L <= 0:
            raise ValueError("L must be >= 1")
        out: list[int] = []
        x = 0x123456789ABCDEF0
        for _ in range(L):
            x, z = cls._splitmix64_next(x)
            c_u32 = z & 0xFFFFFFFF
            if c_u32 == 0:
                c_u32 = 1
            out.append(int(c_u32 - 0x100000000) if c_u32 >= 0x80000000 else int(c_u32))
        return out

    @classmethod
    def sample_params(
        cls,
        B: int,
        device: str | torch.device = "cpu",
        *,
        lambda_: float | tuple[float, float] = 0.5,  # Langton lambda (non-quiescent fraction)
        seed: int | tuple[int, int] | torch.Tensor | None = None,
        seed_range: tuple[int, int] = (0, (1 << 31) - 1),
        compute_actual_lambda: bool = False,
        actual_lambda_system: "LangtonCABase | None" = None,
        actual_lambda_max_neighborhoods: int = 1_000_000,
    ) -> dict:
        """
        Sample random parameters for a Langton CA.

        Args:
            B: Batch size.
            device: Torch device.
            lambda_: Langton lambda value or (lo, hi) range to sample uniformly from.
            seed: Rule seed(s). None -> random, int -> fixed, tuple -> range.
            seed_range: Default range when seed is None.
            compute_actual_lambda: If True, compute the actual (exhaustive) lambda
                and include it in the returned dict as 'actual_lambda'.
            actual_lambda_system: A LangtonCABase instance used for exhaustive
                lambda computation. Required when compute_actual_lambda=True.
            actual_lambda_max_neighborhoods: Max K^L for exhaustive computation.
        """
        dev = torch.device(device)

        # lambda (Langton): fraction non-quiescent
        if isinstance(lambda_, (tuple, list)) and len(lambda_) == 2:
            lo, hi = float(lambda_[0]), float(lambda_[1])
            lam = lo + (hi - lo) * torch.rand(B, device=dev)
        else:
            lam = torch.full((B,), float(lambda_), device=dev)
        lam = lam.clamp(0.0, 1.0).to(torch.float32)

        # seed
        if seed is None:
            lo, hi = int(seed_range[0]), int(seed_range[1])
            seeds = torch.randint(lo, hi + 1, (B,), device=dev, dtype=torch.int64)
        elif torch.is_tensor(seed):
            seeds = seed.to(device=dev, dtype=torch.int64).view(-1)
            if seeds.numel() == 1:
                seeds = seeds.expand(B).contiguous()
            elif seeds.numel() != B:
                raise ValueError(
                    f"seed tensor must have 1 or B elements (got {seeds.numel()} vs B={B})"
                )
        elif isinstance(seed, (tuple, list)) and len(seed) == 2:
            lo, hi = int(seed[0]), int(seed[1])
            seeds = torch.randint(lo, hi + 1, (B,), device=dev, dtype=torch.int64)
        else:
            seeds = torch.full((B,), int(seed), device=dev, dtype=torch.int64)

        params = {"lambda": lam, "seed": seeds}

        if compute_actual_lambda:
            if actual_lambda_system is None:
                raise ValueError(
                    "compute_actual_lambda=True requires actual_lambda_system "
                    "(a LangtonCABase instance) to be provided."
                )
            actual_lam = actual_lambda_system.lambda_actual(
                params,
                max_neighborhoods=actual_lambda_max_neighborhoods,
                warn=True,
            )
            params["actual_lambda"] = actual_lam.to(device=dev, dtype=torch.float32)

        return params

    @classmethod
    def iter_params(
        cls,
        *,
        device: str | torch.device = "cpu",
        batch_size: int = 64,
        **kwargs,
    ):
        while True:
            yield cls.sample_params(batch_size, device=device, **kwargs)

    # -------------------------
    # Hash + rule application
    # -------------------------

    def _infer_device_and_B(self, params: dict) -> tuple[torch.device, int]:
        """
        Infer device/B from params; fallback to self.device and B=1.
        """
        dev = self.device
        B = 1

        seed = params.get("seed", None)
        lam = params.get("lambda", None)

        if torch.is_tensor(seed):
            dev = seed.device
            B = max(B, seed.numel())
        if torch.is_tensor(lam):
            # prefer seed device if present, otherwise lambda device
            if not torch.is_tensor(seed):
                dev = lam.device
            B = max(B, lam.numel())

        return torch.device(dev), int(B)

    def _get_lambda_seed(
        self, params: dict, B: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          lam:  (B,) float32 in [0,1]  (Langton lambda = P(non-quiescent))
          seed: (B,) int64 masked to 32 bits
        """
        lam = params.get("lambda", self._default_lambda)
        if not torch.is_tensor(lam):
            lam = torch.tensor(float(lam), device=device, dtype=torch.float32)
        lam = lam.to(device=device, dtype=torch.float32).view(-1)
        if lam.numel() == 1:
            lam = lam.expand(B)
        elif lam.numel() != B:
            raise ValueError(
                f"params['lambda'] must have 1 or B elements (got {lam.numel()} vs B={B})"
            )
        lam = lam.clamp(0.0, 1.0)

        seed = params.get("seed", self._default_seed)
        if not torch.is_tensor(seed):
            seed = torch.tensor(int(seed), device=device, dtype=torch.int64)
        seed = seed.to(device=device, dtype=torch.int64).view(-1)
        if seed.numel() == 1:
            seed = seed.expand(B)
        elif seed.numel() != B:
            raise ValueError(
                f"params['seed'] must have 1 or B elements (got {seed.numel()} vs B={B})"
            )
        seed = seed & self._MASK32
        return lam, seed

    # ---- safe 32-bit arithmetic helpers ----

    @classmethod
    def _mul32(cls, a: torch.Tensor, b) -> torch.Tensor:
        """
        Compute (a*b) mod 2^32 safely without signed-int64 overflow.

        a: int64 tensor (treated as uint32)
        b: int (python) or int64 tensor (treated as uint32)
        returns: int64 tensor in [0, 2^32-1]
        """
        a = a & cls._MASK32
        if torch.is_tensor(b):
            b = b & cls._MASK32
            b0 = b & 0xFFFF
            b1 = b >> 16
        else:
            b = int(b) & cls._MASK32
            b0 = b & 0xFFFF
            b1 = b >> 16

        a0 = a & 0xFFFF
        a1 = a >> 16

        # (a0*b0 + ((a0*b1 + a1*b0) << 16)) mod 2^32
        res = a0 * b0 + ((a0 * b1 + a1 * b0) << 16)
        return res & cls._MASK32

    @classmethod
    def _fmix32(cls, x: torch.Tensor) -> torch.Tensor:
        """
        MurmurHash3 finalizer on a tensor, treating input as uint32.
        Returns uint32 in int64 tensor [0, 2^32-1].
        """
        x = x & cls._MASK32
        x = x ^ (x >> 16)
        x = cls._mul32(x, cls._FMIX_MUL1)
        x = x ^ (x >> 15)
        x = cls._mul32(x, cls._FMIX_MUL2)
        x = x ^ (x >> 16)
        return x & cls._MASK32

    def _hash31_from_acc(self, acc: torch.Tensor, seed: torch.Tensor) -> torch.Tensor:
        """
        acc:  (B, ...) int64
        seed: (B,)     int64 (masked to 32 bits)
        returns: (B, ...) int64 in [0, 2^31-1]
        """
        # seed_term = (seed * SEED_MUL) mod 2^32
        seed_term_u32 = self._mul32(seed, self._SEED_MUL)  # (B,)
        seed_term = seed_term_u32.view(seed.shape[0], *([1] * (acc.ndim - 1)))

        x = acc + seed_term + self._ACC_OFFSET
        x = self._fmix32(x)
        return x & self._MASK31

    def _states_from_hash(self, h31: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
        """
        h31: (B, ...) int64 in [0, 2^31-1]
        lam: (B,)     float in [0,1], Langton lambda = P(non-quiescent)

        Returns: int64 states in [0, K-1], same shape as h31.
        """
        B = h31.shape[0]
        K = self.num_states
        if K == 1:
            return torch.zeros_like(h31, dtype=torch.int64)

        # Quiescent threshold q = floor((1-lam) * 2^31)
        q_int = torch.floor((1.0 - lam).to(torch.float32) * float(self._MOD31)).to(torch.int64)
        q_int = q_int.clamp(0, self._MOD31)

        view_shape = (B,) + (1,) * (h31.ndim - 1)
        q_b = q_int.view(view_shape)

        is_quiescent = h31 < q_b

        r = (h31 - q_b).clamp(min=0)
        s = r % (K - 1)  # uniform in [0 .. K-2]
        out = torch.where(is_quiescent, torch.zeros_like(r), 1 + s)

        return out.to(torch.int64)

    # -------------------------
    # "Actual lambda" evaluation (exhaustive, when feasible)
    # -------------------------

    def total_neighborhoods(self) -> int:
        """
        Total possible neighborhoods = K^L where:
          K = num_states
          L = neighborhood_len
        """
        if not hasattr(self, "neighborhood_len"):
            raise AttributeError("Subclass must set self.neighborhood_len")
        L = int(self.neighborhood_len)
        return int(pow(int(self.num_states), L))

    @torch.no_grad()
    def _acc_from_basek_indices(self, idx: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Given idx in [0, K^L), interpret idx as base-K digits of length L (flattened neighborhood),
        and compute acc = sum_j digit[j] * coeff_flat[j].

        coeff_flat must be a (L,) int32 buffer.
        Returns:
          acc: (M,) int64
        """
        if not hasattr(self, "_coeff_flat"):
            raise AttributeError("Subclass must register buffer self._coeff_flat (flattened coeffs)")
        if not hasattr(self, "neighborhood_len"):
            raise AttributeError("Subclass must set self.neighborhood_len")

        K = int(self.num_states)
        L = int(self.neighborhood_len)
        coeff = self._coeff_flat.to(device=device, dtype=torch.int64).view(L)

        n = idx.to(device=device, dtype=torch.int64)
        acc = torch.zeros_like(n, dtype=torch.int64)

        # digits for positions L-1..0 via repeated div/mod
        for pos in range(L - 1, -1, -1):
            d = torch.remainder(n, K)
            n = torch.div(n, K, rounding_mode="floor")
            acc = acc + d * coeff[pos]

        return acc

    @torch.no_grad()
    def quiescent_fraction(
        self,
        params: dict,
        *,
        max_neighborhoods: int = 1_000_000,
        chunk_size: int = 262_144,
        warn: bool = True,
    ) -> torch.Tensor:
        """
        Fraction of neighborhoods that map to the quiescent state (state 0).

        This is computed EXACTLY by exhaustively evaluating the rule on all K^L neighborhoods,
        but only when K^L <= max_neighborhoods. Otherwise:
          - emits a warning (optional)
          - returns the *expected* quiescent fraction = 1 - params['lambda']

        Returns:
          qfrac: (B,) float32
        """
        device, B = self._infer_device_and_B(params)
        lam, seed = self._get_lambda_seed(params, B, device)

        # K=1 special case
        if self.num_states == 1:
            return torch.ones(B, device=device, dtype=torch.float32)

        L = int(self.neighborhood_len)
        K = int(self.num_states)

        # Fast cap-check without huge ints
        N = 1
        overflow = False
        for _ in range(L):
            N *= K
            if N > int(max_neighborhoods):
                overflow = True
                break

        if overflow:
            if warn:
                warnings.warn(
                    f"LangtonCA: neighborhood space too large for exhaustive lambda "
                    f"(K^L = {K}^{L} > {max_neighborhoods}). Returning 1-lambda(params) instead.",
                    RuntimeWarning,
                )
            return (1.0 - lam).to(torch.float32)

        N = int(N)  # exact total neighborhoods
        if N == 0:
            return torch.ones(B, device=device, dtype=torch.float32)

        total0 = torch.zeros(B, device=device, dtype=torch.float32)

        chunk_size = max(1, int(chunk_size))
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            idx = torch.arange(start, end, device=device, dtype=torch.int64)  # (M,)

            acc = self._acc_from_basek_indices(idx, device=device)  # (M,)
            acc_b = acc.unsqueeze(0).expand(B, -1)                  # (B,M) (view)

            h31 = self._hash31_from_acc(acc_b, seed)                # (B,M)
            out = self._states_from_hash(h31, lam)                  # (B,M)

            total0 += (out == 0).sum(dim=1, dtype=torch.float32)

        return (total0 / float(N)).to(torch.float32)

    @torch.no_grad()
    def lambda_actual(
        self,
        params: dict,
        *,
        max_neighborhoods: int = 1_000_000,
        chunk_size: int = 262_144,
        warn: bool = True,
    ) -> torch.Tensor:
        """
        Langton's lambda (STANDARD): fraction of NON-QUIESCENT outputs (output != 0).

        Computed exactly when feasible; otherwise returns params['lambda'] (with warning).
        """
        device, B = self._infer_device_and_B(params)
        lam, _ = self._get_lambda_seed(params, B, device)

        L = int(self.neighborhood_len)
        K = int(self.num_states)

        # Fast cap-check
        N = 1
        for _ in range(L):
            N *= K
            if N > int(max_neighborhoods):
                if warn:
                    warnings.warn(
                        f"LangtonCA: neighborhood space too large for exhaustive lambda "
                        f"(K^L = {K}^{L} > {max_neighborhoods}). Returning lambda(params) instead.",
                        RuntimeWarning,
                    )
                return lam.to(torch.float32)

        # Exact: 1 - quiescent_fraction
        q = self.quiescent_fraction(
            params, max_neighborhoods=max_neighborhoods, chunk_size=chunk_size, warn=False
        )
        return (1.0 - q).to(torch.float32)


class LangtonCA1D(LangtonCABase):
    """
    Langton-style 1D CA with K states and implicit random rule table via hashing.

    State:  (B,1,W) integer states in [0, K-1]
    Params: {'lambda': (B,) Langton lambda (non-quiescent fraction), 'seed': (B,)}
    """

    _spatial_ndim = 1

    def __init__(
        self,
        *,
        num_states: int = 4,
        kernel_size: int = 3,
        default_lambda: float = 0.5,
        default_seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        super().__init__(
            num_states=num_states,
            default_lambda=default_lambda,
            default_seed=default_seed,
            device=device,
        )
        self.kernel_size = int(kernel_size)
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be >= 1")
        self.pad_l = (self.kernel_size - 1) // 2
        self.pad_r = self.kernel_size // 2

        self.neighborhood_len = self.kernel_size

        coeff = self._make_coeff_int32(self.kernel_size)
        coeff_t = torch.tensor(coeff, device=self.device, dtype=torch.int32)
        self.register_buffer("_coeff_flat", coeff_t, persistent=False)

    def seed(
        self,
        B: int = 1,
        W: int = 128,
        mode: str = "noise",
        generator=None,
        **kwargs,
    ) -> torch.Tensor:
        if generator is not None:
            return generator(
                B=B, W=W, K=self.num_states, device=self.device, mode=mode, **kwargs
            ).to(self.device)

        mode = str(mode).lower()
        if mode in ("noise", "random"):
            return torch.randint(
                0, self.num_states, (B, 1, W), device=self.device, dtype=torch.int32
            )
        if mode == "zeros":
            return torch.zeros(B, 1, W, device=self.device, dtype=torch.int32)
        if mode == "ones":
            x = torch.zeros(B, 1, W, device=self.device, dtype=torch.int32)
            if self.num_states > 1:
                x.fill_(1)
            return x
        if mode == "single":
            x = torch.zeros(B, 1, W, device=self.device, dtype=torch.int32)
            if self.num_states > 1:
                x[:, 0, W // 2] = 1
            return x
        raise ValueError(f"unknown seed mode: {mode}")

    def forward(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("LangtonCA1D expects x of shape (B,1,W)")
        B, _, _ = x.shape

        # Discretize input to int32 states
        if x.is_floating_point():
            xi = x.round().to(dtype=torch.int32)
        else:
            xi = x.to(dtype=torch.int32)

        if self.num_states > 1:
            xi = xi.clamp(0, self.num_states - 1)
        else:
            xi = torch.zeros_like(xi)

        lam, seed = self._get_lambda_seed(params, B, x.device)

        # Circular pad + unfold neighbourhoods: (B,1,W,K)
        xpad = F.pad(xi, (self.pad_l, self.pad_r), mode="circular")
        patches = xpad.unfold(-1, self.kernel_size, 1)  # view (B,1,W,K)

        coeff = self._coeff_flat.to(device=x.device, dtype=torch.int64)  # (K,)
        acc = (patches.to(torch.int64) * coeff.view(1, 1, 1, -1)).sum(dim=-1).squeeze(1)  # (B,W)

        h31 = self._hash31_from_acc(acc, seed)      # (B,W)
        out = self._states_from_hash(h31, lam)      # (B,W)
        return out.to(dtype=torch.int32).unsqueeze(1)


class LangtonCA2D(LangtonCABase):
    """
    Langton-style 2D CA with K states and implicit random rule table via hashing.

    State:  (B,1,H,W) integer states in [0, K-1]
    Params: {'lambda': (B,) Langton lambda (non-quiescent fraction), 'seed': (B,)}
    """

    _spatial_ndim = 2

    def __init__(
        self,
        *,
        num_states: int = 4,
        kernel_size: int | tuple[int, int] = 3,
        default_lambda: float = 0.5,
        default_seed: int = 0,
        device: str | torch.device = "cpu",
    ):
        super().__init__(
            num_states=num_states,
            default_lambda=default_lambda,
            default_seed=default_seed,
            device=device,
        )

        if isinstance(kernel_size, (tuple, list)):
            KH, KW = int(kernel_size[0]), int(kernel_size[1])
        else:
            KH = KW = int(kernel_size)

        if KH <= 0 or KW <= 0:
            raise ValueError("kernel_size must be >= 1 (or pair of >=1)")
        self.KH = KH
        self.KW = KW

        self.pad_t = (KH - 1) // 2
        self.pad_b = KH // 2
        self.pad_l = (KW - 1) // 2
        self.pad_r = KW // 2

        self.neighborhood_len = KH * KW

        coeff = self._make_coeff_int32(self.neighborhood_len)
        coeff_flat = torch.tensor(coeff, device=self.device, dtype=torch.int32)
        self.register_buffer("_coeff_flat", coeff_flat, persistent=False)

    def seed(
        self,
        B: int = 1,
        H: int = 64,
        W: int = 64,
        mode: str = "noise",
        generator=None,
        **kwargs,
    ) -> torch.Tensor:
        if generator is not None:
            return generator(
                B=B, H=H, W=W, K=self.num_states, device=self.device, mode=mode, **kwargs
            ).to(self.device)

        mode = str(mode).lower()
        if mode in ("noise", "random"):
            return torch.randint(
                0, self.num_states, (B, 1, H, W), device=self.device, dtype=torch.int32
            )
        if mode == "zeros":
            return torch.zeros(B, 1, H, W, device=self.device, dtype=torch.int32)
        if mode == "ones":
            x = torch.zeros(B, 1, H, W, device=self.device, dtype=torch.int32)
            if self.num_states > 1:
                x.fill_(1)
            return x
        if mode == "single":
            x = torch.zeros(B, 1, H, W, device=self.device, dtype=torch.int32)
            if self.num_states > 1:
                x[:, 0, H // 2, W // 2] = 1
            return x
        raise ValueError(f"unknown seed mode: {mode}")

    def forward(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("LangtonCA2D expects x of shape (B,1,H,W)")
        B, _, H, W = x.shape

        # Discretize input to int32 states
        if x.is_floating_point():
            xi = x.round().to(dtype=torch.int32)
        else:
            xi = x.to(dtype=torch.int32)

        if self.num_states > 1:
            xi = xi.clamp(0, self.num_states - 1)
        else:
            xi = torch.zeros_like(xi)

        lam, seed = self._get_lambda_seed(params, B, x.device)

        # Circular pad + unfold neighbourhoods: (B,1,H,W,KH,KW)
        xpad = F.pad(xi, (self.pad_l, self.pad_r, self.pad_t, self.pad_b), mode="circular")
        patches = xpad.unfold(2, self.KH, 1).unfold(3, self.KW, 1)  # view

        # Flatten patch dims and coeffs in the SAME row-major order
        coeff = self._coeff_flat.to(device=x.device, dtype=torch.int64)  # (L,)
        L = self.neighborhood_len
        acc = (patches.to(torch.int64).reshape(B, 1, H, W, L) * coeff.view(1, 1, 1, 1, L)).sum(
            dim=-1
        ).squeeze(1)  # (B,H,W)

        h31 = self._hash31_from_acc(acc, seed)   # (B,H,W)
        out = self._states_from_hash(h31, lam)   # (B,H,W)
        return out.to(dtype=torch.int32).unsqueeze(1)
