// BinaryCA1D: idx = neighborhood bits (leftmost = MSB), next = rule[idx].
struct SimParams {
  B: u32, C: u32, H: u32, W: u32,
  K: u32, numStates: u32, tableLen: u32, flags: u32,
  f0: f32, f1: f32, f2: f32, f3: f32,
};

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read> rule: array<u32>; // (B, 2^K)

@compute @workgroup_size(256, 1, 1)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let b = gid.y;
  if (x >= P.W || b >= P.B) { return; }
  let off = b * P.W;
  let pad = (P.K - 1u) / 2u;
  var idx = 0u;
  for (var p = 0u; p < P.K; p = p + 1u) {
    let xi = (x + P.W - pad + p) % P.W;
    idx = (idx << 1u) | src[off + xi];
  }
  dst[off + x] = rule[b * P.tableLen + idx];
}
