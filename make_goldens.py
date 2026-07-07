"""
Generate golden fixtures for the webapp test suite.

Run from the repo root (needs torch + brotli, same env as the apps):

    uv run python make_goldens.py

Writes webapp/test/fixtures/goldens.json. The JS tests use these to pin
algorithmic parity with the Python implementations (systems bit-exactness,
metric values, brotli byte counts). Regenerate only when the Python
reference implementations change.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from Complexity import (
    CompressedRatio,
    DensityTransientTime,
    Entropy,
    EntropyMinusCompressedRatio,
    FutureStateMutualInformation,
    OpenEndedness,
)
from DynamicalSystems import (
    BinaryCA1D,
    BinaryCA2D,
    CoupledLogistic1D,
    GrayScott2D,
    LangtonCA1D,
    LangtonCA2D,
    OuterTotalisticCA2D,
)
from DynamicalSystems.systems import LangtonCABase
from FeatureExtractor.embedding import Embedding

OUT = Path(__file__).parent / "webapp" / "test" / "fixtures" / "goldens.json"


def rollout_states(system, x0, params, steps):
    """List of state tensors after each of `steps` forward calls."""
    x = x0
    out = []
    with torch.no_grad():
        for _ in range(steps):
            x = system(x, params)
            out.append(x.clone())
    return out


def intlist(t):
    return t.detach().cpu().long().reshape(-1).tolist()


def floatlist(t):
    return [float(v) for v in t.detach().cpu().float().reshape(-1)]


def golden_binary_ca1d():
    W = 16
    system = BinaryCA1D(kernel_size=3)
    params = system.sample_params(1, rule_int=30)
    x0 = system.seed(B=1, W=W, mode="single")
    states = rollout_states(system, x0, params, 8)
    return {"W": W, "rule_int": 30, "x0": intlist(x0), "steps": [intlist(s) for s in states]}


def golden_binary_ca2d():
    H = W = 8
    torch.manual_seed(0)
    system = BinaryCA2D()
    params = system.sample_params(1, p=0.5)
    x0 = torch.zeros(1, 1, H, W)
    for y in range(H):
        for x in range(W):
            if (y * W + x) % 3 == 0:
                x0[0, 0, y, x] = 1.0
    states = rollout_states(system, x0, params, 3)
    return {
        "H": H,
        "W": W,
        "rule": intlist(params["rule"]),
        "x0": intlist(x0),
        "steps": [intlist(s) for s in states],
    }


def golden_ot2d_glider():
    H = W = 8
    system = OuterTotalisticCA2D(kernel_size=3)
    params = system.from_desc("B3/S23", B=1, kernel_size=3)
    x0 = torch.zeros(1, 1, H, W)
    for (y, x) in [(1, 2), (2, 3), (3, 1), (3, 2), (3, 3)]:
        x0[0, 0, y, x] = 1.0
    states = rollout_states(system, x0, params, 4)
    return {"H": H, "W": W, "x0": intlist(x0), "steps": [intlist(s) for s in states]}


def golden_langton_hash():
    inputs = [0, 1, 2, 0xDEADBEEF, 0x7FFFFFFF, 0xFFFFFFFF, 12345678]
    t = torch.tensor(inputs, dtype=torch.int64)
    outputs = [int(v) for v in LangtonCABase._fmix32(t)]
    coeffs = {str(L): LangtonCABase._make_coeff_int32(L) for L in (3, 9, 25)}
    return {"fmix32_inputs": inputs, "fmix32_outputs": outputs, "coeff_signed": coeffs}


def golden_langton1d(lambda_: float, steps: int, seed: int = 12345):
    W, K = 16, 4
    system = LangtonCA1D(num_states=K, kernel_size=3)
    params = {
        "lambda": torch.tensor([lambda_]),
        "seed": torch.tensor([seed], dtype=torch.int64),
    }
    x0 = torch.tensor([[[i % K for i in range(W)]]], dtype=torch.int32)
    states = rollout_states(system, x0, params, steps)
    return {
        "W": W,
        "K": K,
        "kernel": 3,
        "lambda": lambda_,
        "seed": seed,
        "x0": intlist(x0),
        "steps": [intlist(s) for s in states],
    }


def golden_langton2d():
    H = W = 8
    K = 5
    system = LangtonCA2D(num_states=K, kernel_size=3)
    params = {
        "lambda": torch.tensor([0.62]),
        "seed": torch.tensor([987654321], dtype=torch.int64),
    }
    x0 = torch.tensor(
        [[[[(y * W + x) * 7 % K for x in range(W)] for y in range(H)]]], dtype=torch.int32
    )
    states = rollout_states(system, x0, params, 3)
    return {
        "H": H,
        "W": W,
        "K": K,
        "kernel": 3,
        "lambda": 0.62,
        "seed": 987654321,
        "x0": intlist(x0),
        "steps": [intlist(s) for s in states],
    }


def golden_gray_scott():
    H = W = 12
    system = GrayScott2D()
    params = {
        "Du": torch.tensor([0.16]),
        "Dv": torch.tensor([0.08]),
        "F": torch.tensor([0.035]),
        "k": torch.tensor([0.06]),
    }
    x0 = system.seed(B=1, H=H, W=W, noise=0.0)
    states = rollout_states(system, x0, params, 5)
    return {
        "H": H,
        "W": W,
        "Du": 0.16,
        "Dv": 0.08,
        "F": 0.035,
        "k": 0.06,
        "x0": floatlist(x0),
        "steps": [floatlist(s) for s in states],
    }


def golden_coupled_logistic():
    W = 16
    system = CoupledLogistic1D()
    params = {"r": torch.tensor([3.8]), "eps": torch.tensor([0.25])}
    x0 = ((torch.arange(W).float() + 0.5) / W).view(1, 1, W)
    states = rollout_states(system, x0, params, 5)
    return {
        "W": W,
        "r": 3.8,
        "eps": 0.25,
        "x0": floatlist(x0),
        "steps": [floatlist(s) for s in states],
    }


def golden_metric_windows():
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from app_dynamics import metric_windows

    cases = []
    for (T, chunk, stride) in [(10, 4, 2), (5, 0, 1), (8, 8, 1), (8, 12, 3), (6, 2, 5)]:
        cases.append(
            {"T": T, "chunk": chunk, "stride": stride, "windows": metric_windows(T, chunk, stride)}
        )
    return cases


def golden_pca():
    from FeatureExtractor.embedding import _reduce

    T, D = 12, 4
    A = [1.0, 0.5, -0.3, 0.2]
    B_ = [0.2, -1.0, 0.4, 0.7]
    z = torch.zeros(T, D)
    for t in range(T):
        for d in range(D):
            z[t, d] = t * A[d] + torch.sin(torch.tensor(2.1 * t)).item() * B_[d]
    y = _reduce(z, "pca", 2)
    return {"T": T, "D": D, "z": floatlist(z), "y": [float(v) for v in y.reshape(-1)]}


def golden_metrics():
    B, T, D = 2, 16, 6
    disc2 = torch.zeros(B, T, D, dtype=torch.long)
    for b in range(B):
        for t in range(T):
            for d in range(D):
                disc2[b, t, d] = (b * 7 + t * 3 + d) % 2
    disc4 = torch.zeros(B, T, D, dtype=torch.long)
    for b in range(B):
        for t in range(T):
            for d in range(D):
                disc4[b, t, d] = (b + t * 2 + d * d) % 4

    emb2 = Embedding(tensor=disc2, is_discrete=True, num_states=2)
    emb4 = Embedding(tensor=disc4, is_discrete=True, num_states=4)

    cont = torch.zeros(B, T, D)
    for b in range(B):
        for t in range(T):
            for d in range(D):
                cont[b, t, d] = torch.sin(torch.tensor(0.3 * (t + 1) * (d + 1) + b)).item()
    embc = Embedding(tensor=cont, is_discrete=False)

    def run(metric, emb):
        return [float(v) for v in metric(emb)]

    out = {
        "disc2": intlist(disc2),
        "disc4": intlist(disc4),
        "cont": floatlist(cont),
        "B": B,
        "T": T,
        "D": D,
        "entropy_time_k2": run(Entropy("time"), emb2),
        "entropy_space_k2": run(Entropy("space"), emb2),
        "entropy_both_k2": run(Entropy("both"), emb2),
        "entropy_time_k4": run(Entropy("time"), emb4),
        "entropy_both_k4": run(Entropy("both"), emb4),
        "mi_time_k2": run(FutureStateMutualInformation("time"), emb2),
        "mi_space_k2": run(FutureStateMutualInformation("space"), emb2),
        "mi_time_k4": run(FutureStateMutualInformation("time"), emb4),
        "compressed_ratio_k2": run(CompressedRatio(), emb2),
        "compressed_ratio_k4": run(CompressedRatio(), emb4),
        "entropy_minus_cr_k2": run(EntropyMinusCompressedRatio(), emb2),
        "density_transient_k2": run(DensityTransientTime(), emb2),
        "open_endedness_cont": run(OpenEndedness(), embc),
    }
    return out


def main():
    goldens = {
        "binary_ca1d_rule30": golden_binary_ca1d(),
        "binary_ca2d": golden_binary_ca2d(),
        "ot2d_glider": golden_ot2d_glider(),
        "langton_hash": golden_langton_hash(),
        "langton1d_lam50": golden_langton1d(0.5, 5),
        "langton1d_lam37": golden_langton1d(0.37, 3),
        "langton2d": golden_langton2d(),
        "gray_scott": golden_gray_scott(),
        "coupled_logistic": golden_coupled_logistic(),
        "metric_windows": golden_metric_windows(),
        "metrics": golden_metrics(),
        "pca": golden_pca(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(goldens))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
