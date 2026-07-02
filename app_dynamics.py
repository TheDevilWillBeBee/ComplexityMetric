from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
from typing import Callable

import pandas as pd
import torch
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Complexity import (  # noqa: E402
    CompressedRatio,
    DensityTransientTime,
    Entropy,
    EntropyMinusCompressedRatio,
    FutureStateMutualInformation,
    KNNTimeRegression,
    LinearRidgeTimeRegression,
    OpenEndedness,
)
from DynamicalSystems import (  # noqa: E402
    BinaryCA1D,
    BinaryCA2D,
    CoupledLogistic1D,
    GrayScott2D,
    LangtonCA1D,
    LangtonCA2D,
    OuterTotalisticCA1D,
    OuterTotalisticCA2D,
)
from FeatureExtractor import (
    CLIP,
    ContinuousFlatten,
    DiscreteFlatten,
    RandomConvNet,
    RandomVGG,
    SpatialStatistics,
    VGG16,
)  # noqa: E402


@dataclass(frozen=True)
class SystemSpec:
    cls: type
    spatial_dim: int
    default_size: tuple[int | None, int]
    default_steps: tuple[int, int, int]
    discrete: bool
    note: str


@dataclass(frozen=True)
class ExtractorSpec:
    cls: type
    dims: tuple[int, ...]
    note: str


SYSTEMS = {
    "Binary CA 1D": SystemSpec(BinaryCA1D, 1, (None, 128), (128, 1, 0), True, "Elementary binary cellular automata."),
    "Outer-totalistic CA 1D": SystemSpec(OuterTotalisticCA1D, 1, (None, 128), (128, 1, 0), True, "Binary birth/survival rules on a 1D neighborhood."),
    "Binary CA 2D": SystemSpec(BinaryCA2D, 2, (64, 64), (96, 2, 0), True, "Random 3x3 binary rule table."),
    "Outer-totalistic CA 2D": SystemSpec(OuterTotalisticCA2D, 2, (64, 64), (96, 2, 0), True, "Life-like birth/survival cellular automata."),
    "Langton CA 1D": SystemSpec(LangtonCA1D, 1, (None, 128), (128, 1, 0), True, "Implicit random K-state CA controlled by Langton lambda."),
    "Langton CA 2D": SystemSpec(LangtonCA2D, 2, (64, 64), (96, 2, 0), True, "Implicit random K-state 2D CA controlled by Langton lambda."),
    "Coupled logistic map 1D": SystemSpec(CoupledLogistic1D, 1, (None, 256), (256, 2, 32), False, "Continuous coupled chaotic map lattice."),
    "Gray-Scott 2D": SystemSpec(GrayScott2D, 2, (96, 96), (800, 10, 100), False, "Continuous reaction-diffusion system."),
}


EXTRACTORS = {
    "DiscreteFlatten": ExtractorSpec(DiscreteFlatten, (1, 2), "Discrete in, discrete out. Keeps spatial states as raw features."),
    "ContinuousFlatten": ExtractorSpec(ContinuousFlatten, (1, 2), "Continuous in, continuous out. Flattens raw continuous frames."),
    "SpatialStatistics": ExtractorSpec(SpatialStatistics, (1, 2), "Continuous in, continuous out. Per-channel mean/std over spatial axes."),
    "RandomConvNet": ExtractorSpec(RandomConvNet, (1, 2), "Continuous in, continuous out. Stride-1 random convolutions, growing kernels, global reduction."),
    "RandomVGG": ExtractorSpec(RandomVGG, (1, 2), "Continuous in, continuous out. VGG16 block structure with random weights."),
    "VGG16": ExtractorSpec(VGG16, (2,), "Continuous 2D image input, continuous out. Torchvision VGG16 features."),
    "CLIP": ExtractorSpec(CLIP, (2,), "Continuous 2D image input, continuous out. Requires open_clip."),
}


METRICS: dict[str, Callable[[], object]] = {
    "Linear ridge time regression": LinearRidgeTimeRegression,
    "KNN time regression": KNNTimeRegression,
    "Open-endedness": OpenEndedness,
    "Entropy time": lambda: Entropy("time"),
    "Entropy space": lambda: Entropy("space"),
    "Entropy both": lambda: Entropy("both"),
    "CompressedRatio": CompressedRatio,
    "Entropy - CompressedRatio": EntropyMinusCompressedRatio,
    "Density transient time": DensityTransientTime,
    "Future mutual information time": lambda: FutureStateMutualInformation("time"),
    "Future mutual information space": lambda: FutureStateMutualInformation("space"),
}


def device_options() -> list[str]:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def expand_params(params: dict, batch_size: int) -> dict:
    out = {}
    for key, value in params.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == 1:
            out[key] = value.expand(batch_size, *value.shape[1:]).contiguous()
        else:
            out[key] = value
    return out


def sidebar() -> dict:
    st.sidebar.header("Simulation")
    system_name = st.sidebar.selectbox("System", list(SYSTEMS))
    spec = SYSTEMS[system_name]

    device = st.sidebar.selectbox("Device", device_options())
    batch_size = st.sidebar.number_input("Batch", min_value=1, max_value=16, value=4)

    default_h, default_w = spec.default_size
    if spec.spatial_dim == 2:
        c1, c2 = st.sidebar.columns(2)
        H = c1.number_input("Height", min_value=16, max_value=256, value=int(default_h or 64), step=8)
        W = c2.number_input("Width", min_value=16, max_value=256, value=int(default_w), step=8)
    else:
        H = None
        W = st.sidebar.number_input("Width", min_value=16, max_value=1024, value=int(default_w), step=16)

    default_steps, default_every, default_skip = spec.default_steps
    c1, c2, c3 = st.sidebar.columns(3)
    steps = c1.number_input("Steps", min_value=1, max_value=10000, value=default_steps)
    every = c2.number_input("Every", min_value=1, max_value=200, value=default_every)
    skip = c3.number_input("Skip", min_value=0, max_value=int(steps), value=min(default_skip, int(steps)))

    st.sidebar.header("System Parameters")
    cfg = {}
    if system_name == "Binary CA 1D":
        cfg["kernel_size"] = st.sidebar.number_input("Kernel size", min_value=3, max_value=9, value=3, step=2)
        cfg["random_rule"] = st.sidebar.checkbox("Random rule", value=False)
        cfg["rule_int"] = st.sidebar.number_input("Rule", min_value=0, max_value=255, value=30)
    elif system_name in ("Outer-totalistic CA 1D", "Outer-totalistic CA 2D"):
        cfg["kernel_size"] = st.sidebar.number_input("Kernel size", min_value=3, max_value=11, value=3 if spec.spatial_dim == 2 else 5, step=2)
        cfg["random_rule"] = st.sidebar.checkbox("Random rule", value=False)
        cfg["desc"] = st.sidebar.text_input("Rule", value="B3/S23" if spec.spatial_dim == 2 else "B1/S23")
    elif system_name == "Binary CA 2D":
        cfg["p"] = st.sidebar.slider("Rule density", 0.0, 1.0, 0.5, 0.01)
    elif system_name.startswith("Langton"):
        cfg["num_states"] = st.sidebar.number_input("States", min_value=2, max_value=16, value=4 if spec.spatial_dim == 2 else 2)
        cfg["kernel_size"] = st.sidebar.number_input("Kernel size", min_value=1, max_value=7, value=3, step=2)
        cfg["lambda"] = st.sidebar.slider("Lambda", 0.0, 1.0, 0.5, 0.01)
        cfg["rule_seed"] = st.sidebar.number_input("Rule seed", min_value=0, max_value=(1 << 31) - 1, value=0)
    elif system_name == "Coupled logistic map 1D":
        cfg["r"] = st.sidebar.slider("r", 3.0, 4.0, 3.8, 0.01)
        cfg["eps"] = st.sidebar.slider("epsilon", 0.0, 1.0, 0.25, 0.01)
    elif system_name == "Gray-Scott 2D":
        cfg["Du"] = st.sidebar.number_input("Du", min_value=0.0, max_value=1.0, value=0.16, format="%.4f")
        cfg["Dv"] = st.sidebar.number_input("Dv", min_value=0.0, max_value=1.0, value=0.08, format="%.4f")
        cfg["F"] = st.sidebar.number_input("F", min_value=0.0, max_value=0.2, value=0.035, format="%.4f")
        cfg["k"] = st.sidebar.number_input("k", min_value=0.0, max_value=0.2, value=0.060, format="%.4f")

    st.sidebar.header("Initial State")
    seed_modes = ["noise", "single", "zeros"] if spec.discrete else ["noise"]
    cfg["seed_mode"] = st.sidebar.selectbox("Seed mode", seed_modes)
    if spec.discrete:
        cfg["seed_p"] = st.sidebar.slider("Initial density", 0.0, 1.0, 0.5, 0.01)

    st.sidebar.header("Feature Extractors")
    options = [
        name
        for name, ext in EXTRACTORS.items()
        if spec.spatial_dim in ext.dims
        and (name != "CLIP" or clip_available())
        and (name != "VGG16" or vgg16_available())
    ]
    selected_extractors = st.sidebar.multiselect("Extractors", options, default=["DiscreteFlatten" if spec.discrete else "RandomConvNet"])
    if not selected_extractors:
        selected_extractors = ["DiscreteFlatten" if spec.discrete else "RandomConvNet"]

    with st.sidebar.expander("Extractor info", expanded=True):
        st.dataframe(extractor_info(spec.spatial_dim), use_container_width=True, hide_index=True)

    st.sidebar.header("Embedding")
    dim_method = st.sidebar.selectbox("Trajectory projection", ["pca", "tsne", "umap"])
    embed_dim = st.sidebar.number_input("Feature dimension", min_value=2, max_value=512, value=64, step=8)
    extractor_seed = st.sidebar.number_input("Extractor seed", min_value=0, max_value=(1 << 31) - 1, value=0)
    vgg16_pretrained = st.sidebar.checkbox("VGG16 ImageNet weights", value=False)

    st.sidebar.header("Metrics")
    selected_metrics = st.sidebar.multiselect("Metrics", list(METRICS), default=["Linear ridge time regression", "Entropy both", "CompressedRatio"])
    metric_chunks_enabled = st.sidebar.checkbox("Temporal metric slices", value=False)
    metric_chunk_size = st.sidebar.number_input(
        "Metric chunk size",
        min_value=0,
        value=0,
        step=1,
        help="Use 0 to score the full embedding sequence.",
    )
    metric_stride = st.sidebar.number_input("Metric stride", min_value=1, value=1, step=1)
    with st.sidebar.expander("Metric input types"):
        st.dataframe(metric_info(), use_container_width=True, hide_index=True)

    st.sidebar.header("Randomness")
    torch_seed = st.sidebar.number_input("Torch seed", min_value=0, max_value=(1 << 31) - 1, value=0)

    return {
        "system_name": system_name,
        "device": device,
        "batch_size": int(batch_size),
        "H": None if H is None else int(H),
        "W": int(W),
        "steps": int(steps),
        "every": int(every),
        "skip": int(skip),
        "extractors": selected_extractors,
        "metrics": selected_metrics,
        "dim_method": dim_method,
        "embed_dim": int(embed_dim),
        "extractor_seed": int(extractor_seed),
        "vgg16_pretrained": bool(vgg16_pretrained),
        "torch_seed": int(torch_seed),
        "metric_chunks_enabled": bool(metric_chunks_enabled),
        "metric_chunk_size": int(metric_chunk_size),
        "metric_stride": int(metric_stride),
        **cfg,
    }


def clip_available() -> bool:
    try:
        import open_clip  # noqa: F401
    except Exception:
        return False
    return True


def vgg16_available() -> bool:
    try:
        import torchvision  # noqa: F401
    except Exception:
        return False
    return True


def extractor_info(spatial_dim: int) -> pd.DataFrame:
    rows = []
    for name, spec in EXTRACTORS.items():
        available = (
            spatial_dim in spec.dims
            and (name != "CLIP" or clip_available())
            and (name != "VGG16" or vgg16_available())
        )
        rows.append(
            {
                "extractor": name,
                "input": spec.cls.input_type,
                "output": spec.cls.output_type,
                "spatial": ", ".join(f"{d}D" for d in spec.dims),
                "available": available,
                "note": spec.note,
            }
        )
    return pd.DataFrame(rows)


def metric_info() -> pd.DataFrame:
    rows = []
    for name, factory in METRICS.items():
        metric = factory()
        rows.append({"metric": name, "expects": metric.input_type})
    return pd.DataFrame(rows)


def make_system(cfg: dict):
    name = cfg["system_name"]
    device = cfg["device"]
    if name == "Binary CA 1D":
        return BinaryCA1D(kernel_size=cfg["kernel_size"], device=device)
    if name == "Outer-totalistic CA 1D":
        return OuterTotalisticCA1D(kernel_size=cfg["kernel_size"], device=device)
    if name == "Outer-totalistic CA 2D":
        return OuterTotalisticCA2D(kernel_size=cfg["kernel_size"], device=device)
    if name == "Langton CA 1D":
        return LangtonCA1D(num_states=cfg["num_states"], kernel_size=cfg["kernel_size"], default_lambda=cfg["lambda"], device=device)
    if name == "Langton CA 2D":
        return LangtonCA2D(num_states=cfg["num_states"], kernel_size=cfg["kernel_size"], default_lambda=cfg["lambda"], device=device)
    if name == "Gray-Scott 2D":
        return GrayScott2D(device=device)
    return SYSTEMS[name].cls(device=device)


def make_params(system, cfg: dict) -> dict:
    B, device, name = cfg["batch_size"], cfg["device"], cfg["system_name"]
    if name == "Binary CA 1D":
        params = system.sample_params(1, device=device, kernel_size=cfg["kernel_size"], rule_int=None if cfg["random_rule"] else cfg["rule_int"])
    elif name in ("Outer-totalistic CA 1D", "Outer-totalistic CA 2D"):
        if cfg["random_rule"]:
            params = system.sample_params(1, device=device, kernel_size=cfg["kernel_size"])
        else:
            params = system.from_desc(cfg["desc"], B=1, kernel_size=cfg["kernel_size"], device=device)
    elif name == "Binary CA 2D":
        params = system.sample_params(1, device=device, p=cfg["p"])
    elif name.startswith("Langton"):
        params = system.sample_params(1, device=device, lambda_=cfg["lambda"], seed=int(cfg["rule_seed"]))
    elif name == "Coupled logistic map 1D":
        params = {"r": torch.full((1,), cfg["r"], device=device), "eps": torch.full((1,), cfg["eps"], device=device)}
    elif name == "Gray-Scott 2D":
        params = {k: torch.full((1,), cfg[k], device=device) for k in ("Du", "Dv", "F", "k")}
    else:
        params = system.sample_params(1, device=device)
    return expand_params(params, B)


def make_seed(system, cfg: dict):
    B, W, H = cfg["batch_size"], cfg["W"], cfg["H"]
    mode = cfg["seed_mode"]
    if cfg["system_name"] == "Gray-Scott 2D":
        return system.seed(B=B, H=H, W=W)
    if cfg["system_name"] == "Coupled logistic map 1D":
        return system.seed(B=B, W=W, mode=mode)
    if H is None:
        return system.seed(B=B, W=W, mode=mode, p=cfg.get("seed_p", 0.5))
    return system.seed(B=B, H=H, W=W, mode=mode, p=cfg.get("seed_p", 0.5))


def make_extractor(name: str, rollout, cfg: dict):
    common = {"spatial_dim": rollout.spatial_dim, "device": cfg["device"]}
    if name == "DiscreteFlatten":
        return DiscreteFlatten(**common).eval(), 0
    if name == "ContinuousFlatten":
        return ContinuousFlatten(**common).eval(), 0
    if name == "SpatialStatistics":
        return SpatialStatistics(**common).eval(), 0
    if name == "RandomConvNet":
        return (
            RandomConvNet(
                **common,
                embed_dim=cfg["embed_dim"],
                base_channels=8,
                num_stages=3,
                layers_per_stage=2,
                reduction="gram",
                activation="tanh",
                norm="instance",
                use_pooling=False,
                kernel_size=5,
                kernel_growth=2,
                seed=cfg["extractor_seed"],
            ).eval(),
            64,
        )
    if name == "RandomVGG":
        return RandomVGG(**common, embed_dim=cfg["embed_dim"], channels=(8, 16, 32), seed=cfg["extractor_seed"]).eval(), 64
    if name == "VGG16":
        return (
            VGG16(
                embed_dim=cfg["embed_dim"],
                pretrained=cfg["vgg16_pretrained"],
                seed=cfg["extractor_seed"],
                device=cfg["device"],
            ).eval(),
            16,
        )
    if name == "CLIP":
        return CLIP(embed_dim=cfg["embed_dim"], device=cfg["device"]).eval(), 16
    raise ValueError(f"unknown extractor: {name}")


def metric_windows(T: int, chunk_size: int, stride: int) -> list[tuple[int, int]]:
    if T <= 0:
        return []
    size = T if int(chunk_size) <= 0 else min(int(chunk_size), T)
    step = max(1, int(stride))
    return [(start, start + size) for start in range(0, T - size + 1, step)]


def compute_metrics(
    embedding,
    names: list[str],
    *,
    chunk_size: int = 0,
    stride: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    windows = metric_windows(embedding.T, chunk_size, stride)
    rows = []
    curve_rows = []
    for name in names:
        metric = METRICS[name]()
        try:
            chunk_scores = []
            chunk_means = []
            metric_curve_rows = []
            for chunk_idx, (start, end) in enumerate(windows):
                scores = metric(embedding[start:end]).detach().cpu().float()
                chunk_scores.append(scores)
                mean = float(scores.mean())
                chunk_means.append(mean)
                metric_curve_rows.append(
                    {
                        "metric": name,
                        "chunk": chunk_idx,
                        "start": start,
                        "end": end,
                        "time": start + (end - start - 1) / 2.0,
                        "mean": mean,
                        "std": float(scores.std()) if scores.numel() > 1 else 0.0,
                    }
                )

            scores = torch.cat(chunk_scores) if chunk_scores else torch.empty(0)
            curve_rows.extend(metric_curve_rows)
            rows.append(
                {
                    "metric": name,
                    "expects": metric.input_type,
                    "chunks": len(windows),
                    "chunk_size": 0 if not windows else windows[0][1] - windows[0][0],
                    "stride": max(1, int(stride)),
                    "mean": float(scores.mean()),
                    "std": float(scores.std()) if scores.numel() > 1 else 0.0,
                    "chunk_means": ", ".join(f"{v:.4f}" for v in chunk_means),
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "metric": name,
                    "expects": metric.input_type,
                    "chunks": len(windows),
                    "chunk_size": 0 if not windows else windows[0][1] - windows[0][0],
                    "stride": max(1, int(stride)),
                    "mean": None,
                    "std": None,
                    "chunk_means": "",
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(curve_rows)


def show_rollout(rollout):
    st.subheader("Rollout")
    st.write(
        {
            "shape": list(rollout.tensor.shape),
            "discrete": rollout.is_discrete,
            "num_states": rollout.num_states,
            "spatial_dim": rollout.spatial_dim,
            "steps": rollout.steps,
            "every": rollout.every,
            "skip": rollout.skip,
        }
    )
    st.image(rollout.visualize(t=-1), use_container_width=True)
    if rollout.spatial_dim == 2:
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.close()
            st.video(rollout.visualize_video(filename=tmp.name, fps=24, every=1))
        except Exception as exc:
            st.warning(f"Video failed: {exc}")


def run(cfg: dict):
    torch.manual_seed(cfg["torch_seed"])
    system = make_system(cfg)
    params = make_params(system, cfg)
    x0 = make_seed(system, cfg)

    with torch.no_grad():
        rollout = system.rollout(x0, params, steps=cfg["steps"], every=cfg["every"], skip=cfg["skip"])

    show_rollout(rollout)

    st.subheader("Feature Extractors")
    tabs = st.tabs(cfg["extractors"])
    for tab, name in zip(tabs, cfg["extractors"]):
        with tab:
            extractor, chunk = make_extractor(name, rollout, cfg)
            with torch.no_grad():
                embedding = extractor(rollout, chunk=chunk)

            st.write(
                {
                    "input_type": extractor.input_type,
                    "output_type": extractor.output_type,
                    "embedding_shape": list(embedding.tensor.shape),
                    "embedding_discrete": embedding.is_discrete,
                    "num_states": embedding.num_states,
                }
            )
            try:
                st.image(embedding.visualize(method=cfg["dim_method"]), use_container_width=True)
            except Exception as exc:
                st.warning(f"Embedding visualization failed: {exc}")

            if cfg["metrics"]:
                chunk_size = cfg["metric_chunk_size"] if cfg["metric_chunks_enabled"] else 0
                stride = cfg["metric_stride"] if cfg["metric_chunks_enabled"] else 1
                metric_df, curve_df = compute_metrics(
                    embedding,
                    cfg["metrics"],
                    chunk_size=chunk_size,
                    stride=stride,
                )
                st.dataframe(metric_df, use_container_width=True, hide_index=True)
                if cfg["metric_chunks_enabled"] and not curve_df.empty:
                    chart = curve_df.pivot(index="time", columns="metric", values="mean")
                    st.line_chart(chart)


def main():
    st.set_page_config(page_title="Dynamical Systems Explorer", layout="wide")
    st.title("Dynamical Systems Explorer")
    st.caption("Refactored app using Rollout, Embedding, FeatureExtractor, and ComplexityMetric objects from src.")
    cfg = sidebar()

    spec = SYSTEMS[cfg["system_name"]]
    st.info(f"{cfg['system_name']}: {spec.note}")
    st.dataframe(extractor_info(spec.spatial_dim), use_container_width=True, hide_index=True)

    if st.button("Run simulation", type="primary"):
        run(cfg)


if __name__ == "__main__":
    main()
