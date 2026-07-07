# Dynamical Systems Explorer (webapp)

Standalone browser port of `app_dynamics.py` — simulation, feature
extraction, trajectory projection, and complexity metrics all run locally
in the browser. No Python backend.

## Run

```bash
cd webapp
npm install
npm run dev        # dev server; open the printed URL
npm run build      # static bundle in dist/ (serve with any static server)
```

## Architecture

- **Systems** (`src/systems/`): the 8 dynamical systems from
  `DynamicalSystems/systems.py`, each with a plain-TypeScript CPU stepper
  (`cpu/`) and a WebGPU compute-shader stepper (`gpu/`, WGSL in
  `gpu/shaders/`). WebGPU is used automatically when available; discrete
  systems are bit-exact across both backends.
- **Extractors** (`src/extractors/`): DiscreteFlatten, ContinuousFlatten,
  SpatialStatistics (pure TS) and RandomConvNet / RandomVGG (TensorFlow.js,
  backend chain webgpu → webgl → cpu). Pretrained VGG16/CLIP are not ported.
- **Metrics** (`src/metrics/`): the 8 complexity metrics from
  `Complexity/`, in TypeScript; CompressedRatio uses `brotli-wasm` at
  quality 11 for byte-level parity with Python `brotli`.
- **Projections** (`src/projection/`): PCA (built-in), t-SNE (`tsne-js`,
  not seedable), UMAP (`umap-js`, seeded).

Seeds are deterministic **within the app** but do not reproduce PyTorch's
RNG — random rule tables and initial states differ from the Python app at
equal seeds. Algorithms are pinned to the Python reference by golden
fixtures instead.

## Tests & parity

```bash
npm test                 # vitest: ground truths + Python-parity goldens
```

Golden fixtures come from the Python reference implementations. Generate
them from the repo root (needs the repo's torch environment):

```bash
uv run python make_goldens.py
```

Until the fixture file exists, the parity tests skip with a warning.

CPU ↔ WebGPU parity: run `npm run dev` and open `/dev/gpu-parity.html` in a
WebGPU-capable browser (bit-equality for the six discrete systems, float32
tolerance for Gray-Scott and the coupled logistic map, plus a Langton-2D
benchmark).
