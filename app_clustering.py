from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd
import torch
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DynamicalSystems import BinaryCA1D, Rollout  # noqa: E402
from FeatureExtractor import (  # noqa: E402
    ContinuousFlatten,
    DiscreteFlatten,
    RandomConvNet,
    RandomVGG,
    SpatialStatistics,
)


@dataclass(frozen=True)
class ExtractorSpec:
    cls: type
    note: str


EXTRACTORS = {
    "DiscreteFlatten": ExtractorSpec(DiscreteFlatten, "Discrete Hamming features from final states."),
    "ContinuousFlatten": ExtractorSpec(ContinuousFlatten, "One-hot continuous flattening of final states."),
    "SpatialStatistics": ExtractorSpec(SpatialStatistics, "Mean/std summary over the final 1D lattice."),
    "RandomConvNet": ExtractorSpec(RandomConvNet, "Random convolutional continuous embedding."),
    "RandomVGG": ExtractorSpec(RandomVGG, "VGG16-style random convolutional continuous embedding."),
}


def device_options() -> list[str]:
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@st.cache_data
def canonical_eca_rules() -> list[int]:
    return sorted({min(_eca_orbit(rule)) for rule in range(256)})


def _eca_orbit(rule: int) -> set[int]:
    seen = set()
    stack = [int(rule)]
    while stack:
        r = stack.pop()
        if r in seen:
            continue
        seen.add(r)
        stack += [_reflect_rule(r), _complement_rule(r)]
    return seen


def _rule_bits(rule: int) -> list[int]:
    return [(int(rule) >> i) & 1 for i in range(8)]


def _rule_from_bits(bits: list[int]) -> int:
    return sum(int(v) << i for i, v in enumerate(bits))


def _reflect_rule(rule: int) -> int:
    bits = _rule_bits(rule)
    out = [0] * 8
    for idx in range(8):
        left = (idx >> 2) & 1
        center = (idx >> 1) & 1
        right = idx & 1
        reflected = (right << 2) | (center << 1) | left
        out[idx] = bits[reflected]
    return _rule_from_bits(out)


def _complement_rule(rule: int) -> int:
    bits = _rule_bits(rule)
    return _rule_from_bits([1 - bits[idx ^ 7] for idx in range(8)])


def sidebar() -> dict:
    st.sidebar.header("Simulation")
    cfg = {
        "device": st.sidebar.selectbox("Device", device_options()),
        "steps": int(st.sidebar.number_input("K steps", min_value=1, max_value=5000, value=128, step=8)),
        "batch_size": int(st.sidebar.number_input("B per rule", min_value=1, max_value=32, value=4)),
        "width": int(st.sidebar.number_input("Width", min_value=32, max_value=2048, value=256, step=32)),
        "seed_mode": st.sidebar.selectbox("Seed mode", ["noise", "single", "zeros"]),
        "seed_p": float(st.sidebar.slider("Initial density", 0.0, 1.0, 0.5, 0.01)),
        "torch_seed": int(st.sidebar.number_input("Torch seed", min_value=0, max_value=(1 << 31) - 1, value=0)),
    }

    st.sidebar.header("Features")
    cfg["extractor"] = st.sidebar.selectbox("Feature extractor", list(EXTRACTORS), index=0)
    cfg["embed_dim"] = int(st.sidebar.number_input("Feature dimension", min_value=2, max_value=512, value=64, step=8))
    cfg["extractor_seed"] = int(st.sidebar.number_input("Extractor seed", min_value=0, max_value=(1 << 31) - 1, value=0))
    cfg["embed_chunk"] = int(st.sidebar.number_input("Embedding chunk", min_value=0, max_value=4096, value=256, step=64))

    st.sidebar.header("Clustering")
    cfg["n_clusters"] = int(st.sidebar.number_input("Clusters", min_value=2, max_value=88, value=8))
    cfg["cluster_iters"] = int(st.sidebar.number_input("K-medoids iterations", min_value=1, max_value=100, value=20))

    return cfg


def make_extractor(name: str, cfg: dict):
    common = {"spatial_dim": 1, "device": cfg["device"]}
    if name == "DiscreteFlatten":
        return DiscreteFlatten(**common).eval()
    if name == "ContinuousFlatten":
        return ContinuousFlatten(**common).eval()
    if name == "SpatialStatistics":
        return SpatialStatistics(**common).eval()
    if name == "RandomConvNet":
        return RandomConvNet(**common, embed_dim=cfg["embed_dim"], seed=cfg["extractor_seed"]).eval()
    if name == "RandomVGG":
        return RandomVGG(**common, embed_dim=cfg["embed_dim"], seed=cfg["extractor_seed"]).eval()
    raise ValueError(f"unknown extractor: {name}")


def simulate_final_states(cfg: dict, rules: list[int]) -> tuple[Rollout, torch.Tensor]:
    device = torch.device(cfg["device"])
    torch.manual_seed(cfg["torch_seed"])
    system = BinaryCA1D(kernel_size=3, device=device)
    rule_ids = torch.tensor(rules, device=device, dtype=torch.long).repeat_interleave(cfg["batch_size"])
    rule_table = ((rule_ids[:, None] >> torch.arange(8, device=device)) & 1).float()
    x = system.seed(
        B=rule_ids.numel(),
        W=cfg["width"],
        mode=cfg["seed_mode"],
        p=cfg["seed_p"],
    )
    params = {"rule": rule_table}
    with torch.no_grad():
        for _ in range(cfg["steps"]):
            x = system(x, params)
    rollout = Rollout(
        tensor=x.unsqueeze(1),
        steps=cfg["steps"],
        every=1,
        skip=0,
        is_discrete=True,
        num_states=2,
        spatial_dim=1,
        to_rgb_fn=system.to_rgb,
        system_name="BinaryCA1D",
    )
    return rollout, rule_ids.cpu()


def embed_final_states(rollout: Rollout, cfg: dict):
    extractor = make_extractor(cfg["extractor"], cfg)
    with torch.no_grad():
        embedding = extractor(rollout, chunk=cfg["embed_chunk"])
    return embedding, extractor


def pairwise_distances(z: torch.Tensor, *, discrete: bool, chunk: int = 512) -> torch.Tensor:
    z = z.detach()
    n = z.shape[0]
    out = torch.empty(n, n, device=z.device, dtype=torch.float32)
    chunk = max(1, int(chunk))
    if discrete:
        y = z.long()
        for start in range(0, n, chunk):
            out[start : start + chunk] = (y[start : start + chunk, None] != y[None]).float().mean(dim=-1)
        return out
    y = z.float()
    for start in range(0, n, chunk):
        out[start : start + chunk] = torch.cdist(y[start : start + chunk], y)
    return out


def distance_summary(dist: torch.Tensor, rule_ids: torch.Tensor) -> pd.DataFrame:
    same = rule_ids[:, None] == rule_ids[None, :]
    upper = torch.triu(torch.ones_like(same, dtype=torch.bool), diagonal=1)
    rows = []
    for name, mask in (("intra_rule", same & upper), ("inter_rule", (~same) & upper)):
        values = dist.cpu()[mask]
        rows.append(
            {
                "comparison": name,
                "pairs": int(values.numel()),
                "mean": float(values.mean()) if values.numel() else None,
                "median": float(values.median()) if values.numel() else None,
                "std": float(values.std(unbiased=False)) if values.numel() else None,
            }
        )
    return pd.DataFrame(rows)


def k_medoids(dist: torch.Tensor, n_clusters: int, *, max_iter: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    n = dist.shape[0]
    k = min(int(n_clusters), n)
    generator = torch.Generator(device=dist.device).manual_seed(int(seed))
    medoids = torch.randperm(n, generator=generator, device=dist.device)[:k]
    labels = torch.zeros(n, device=dist.device, dtype=torch.long)
    for _ in range(int(max_iter)):
        labels = dist[:, medoids].argmin(dim=1)
        new_medoids = medoids.clone()
        for cluster in range(k):
            members = torch.where(labels == cluster)[0]
            if members.numel() == 0:
                nearest = dist[:, medoids].amin(dim=1)
                new_medoids[cluster] = nearest.argmax()
                continue
            within = dist[members][:, members]
            new_medoids[cluster] = members[within.sum(dim=1).argmin()]
        if torch.equal(new_medoids, medoids):
            break
        medoids = new_medoids
    return dist[:, medoids].argmin(dim=1).cpu(), medoids.cpu()


def composition_tables(labels: torch.Tensor, rule_ids: torch.Tensor, rules: list[int], batch_size: int):
    clusters = sorted(int(v) for v in labels.unique().tolist())
    matrix = pd.DataFrame(0.0, index=[f"cluster_{c}" for c in clusters], columns=[str(r) for r in rules])
    rows = []
    for cluster in clusters:
        mask = labels == cluster
        cluster_size = int(mask.sum())
        for rule in rules:
            count = int(((rule_ids == rule) & mask).sum())
            pct_rule = 100.0 * count / float(batch_size)
            matrix.loc[f"cluster_{cluster}", str(rule)] = pct_rule
            if count:
                rows.append(
                    {
                        "cluster": cluster,
                        "rule": rule,
                        "count": count,
                        "pct_of_rule_batch": pct_rule,
                        "pct_of_cluster": 100.0 * count / float(cluster_size),
                    }
                )
    long = pd.DataFrame(rows).sort_values(["cluster", "pct_of_cluster"], ascending=[True, False])
    return matrix.reset_index(names="cluster"), long


def run(cfg: dict):
    rules = canonical_eca_rules()
    rollout, rule_ids = simulate_final_states(cfg, rules)
    embedding, extractor = embed_final_states(rollout, cfg)
    z = embedding.tensor[:, 0].detach()
    dist = pairwise_distances(z, discrete=embedding.is_discrete, chunk=512).cpu()
    labels, medoids = k_medoids(dist, cfg["n_clusters"], max_iter=cfg["cluster_iters"], seed=cfg["torch_seed"])
    composition, cluster_rules = composition_tables(labels, rule_ids, rules, cfg["batch_size"])

    st.subheader("Run Summary")
    st.write(
        {
            "rules": len(rules),
            "points": int(z.shape[0]),
            "steps_per_rule": cfg["steps"],
            "batch_per_rule": cfg["batch_size"],
            "final_rollout_shape": list(rollout.tensor.shape),
            "extractor": type(extractor).__name__,
            "embedding_shape": list(embedding.tensor.shape),
            "embedding_discrete": embedding.is_discrete,
            "distance": "normalized Hamming" if embedding.is_discrete else "L2",
        }
    )

    st.subheader("Inter vs Intra Rule Distance")
    st.dataframe(distance_summary(dist, rule_ids), use_container_width=True, hide_index=True)

    st.subheader("Clusters")
    medoid_rules = rule_ids[medoids].tolist()
    st.dataframe(
        pd.DataFrame({"cluster": list(range(len(medoids))), "medoid_index": medoids.tolist(), "medoid_rule": medoid_rules}),
        use_container_width=True,
        hide_index=True,
    )
    st.dataframe(composition, use_container_width=True, hide_index=True)

    st.subheader("Cluster Rule Contents")
    st.dataframe(cluster_rules, use_container_width=True, hide_index=True)

    with st.expander("Canonical ECA rules"):
        st.write(rules)


def main():
    st.set_page_config(page_title="ECA Rule Clustering", layout="wide")
    st.title("ECA Rule Clustering")
    st.caption("Final-state embeddings for the 88 elementary cellular automata rules modulo reflection and complement symmetries.")
    cfg = sidebar()
    if st.button("Run clustering", type="primary"):
        run(cfg)


if __name__ == "__main__":
    main()
