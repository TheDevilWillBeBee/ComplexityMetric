// OuterTotalisticCA2D: KxK neighbor sum excluding center; birth/survive.
// bs layout: birth at [b*2L + n], survive at [b*2L + L + n], L = tableLen = K*K.
struct SimParams {
  B: u32, C: u32, H: u32, W: u32,
  K: u32, numStates: u32, tableLen: u32, flags: u32,
  f0: f32, f1: f32, f2: f32, f3: f32,
};

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read> bs: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let b = gid.z;
  if (x >= P.W || y >= P.H || b >= P.B) { return; }
  let off = b * P.H * P.W;
  let pad = (P.K - 1u) / 2u;
  var nsum = 0u;
  for (var ky = 0u; ky < P.K; ky = ky + 1u) {
    let yy = ((y + P.H - pad + ky) % P.H) * P.W;
    for (var kx = 0u; kx < P.K; kx = kx + 1u) {
      nsum = nsum + src[off + yy + (x + P.W - pad + kx) % P.W];
    }
  }
  let c = src[off + y * P.W + x];
  nsum = nsum - c;
  let L = P.tableLen;
  let base = b * 2u * L;
  dst[off + y * P.W + x] = select(bs[base + nsum], bs[base + L + nsum], c > 0u);
}
