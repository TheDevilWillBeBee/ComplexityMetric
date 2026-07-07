// LangtonCA2D: same hash pipeline as 1D over a KxK window; coefficient
// index is row-major over the window (matches torch unfold order).
struct SimParams {
  B: u32, C: u32, H: u32, W: u32,
  K: u32, numStates: u32, tableLen: u32, flags: u32,
  f0: f32, f1: f32, f2: f32, f3: f32,
};

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read> coeff: array<u32>; // (K*K,)
@group(0) @binding(4) var<storage, read> aux: array<u32>;   // (B, 2)

fn fmix32(h0: u32) -> u32 {
  var h = h0;
  h = h ^ (h >> 16u);
  h = h * 0x7feb352du;
  h = h ^ (h >> 15u);
  h = h * 0x846ca68bu;
  h = h ^ (h >> 16u);
  return h;
}

@compute @workgroup_size(8, 8, 1)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let b = gid.z;
  if (x >= P.W || y >= P.H || b >= P.B) { return; }
  let off = b * P.H * P.W;
  let pad = (P.K - 1u) / 2u;
  var acc = 0u;
  var j = 0u;
  for (var ky = 0u; ky < P.K; ky = ky + 1u) {
    let yy = ((y + P.H - pad + ky) % P.H) * P.W;
    for (var kx = 0u; kx < P.K; kx = kx + 1u) {
      acc = acc + src[off + yy + (x + P.W - pad + kx) % P.W] * coeff[j];
      j = j + 1u;
    }
  }
  let h = fmix32(acc + aux[b * 2u] + 0x85ebca6bu) & 0x7fffffffu;
  let q = aux[b * 2u + 1u];
  var out = 0u;
  if (h >= q) {
    out = 1u + (h - q) % (P.numStates - 1u);
  }
  dst[off + y * P.W + x] = out;
}
