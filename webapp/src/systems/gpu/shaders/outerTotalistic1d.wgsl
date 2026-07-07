// OuterTotalisticCA1D: neighbor sum excluding center; birth/survive tables.
// bs layout: birth at [b*2L + n], survive at [b*2L + L + n], L = tableLen.
struct SimParams {
  B: u32, C: u32, H: u32, W: u32,
  K: u32, numStates: u32, tableLen: u32, flags: u32,
  f0: f32, f1: f32, f2: f32, f3: f32,
};

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read> bs: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let b = gid.y;
  if (x >= P.W || b >= P.B) { return; }
  let off = b * P.W;
  let pad = (P.K - 1u) / 2u;
  var nsum = 0u;
  for (var p = 0u; p < P.K; p = p + 1u) {
    nsum = nsum + src[off + (x + P.W - pad + p) % P.W];
  }
  let c = src[off + x];
  nsum = nsum - c;
  let L = P.tableLen;
  let base = b * 2u * L;
  dst[off + x] = select(bs[base + nsum], bs[base + L + nsum], c > 0u);
}
