// BinaryCA2D: 3x3 neighborhood -> 9-bit index with the EXACT bit weights of
// the Python implementation (systems.py:449-453 after flip+cross-correlation):
//   (y-1): 64 128 256 / (y): 8 16 32 (center 16) / (y+1): 1 2 4
struct SimParams {
  B: u32, C: u32, H: u32, W: u32,
  K: u32, numStates: u32, tableLen: u32, flags: u32,
  f0: f32, f1: f32, f2: f32, f3: f32,
};

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read> rule: array<u32>; // (B, 512)

@compute @workgroup_size(8, 8, 1)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let b = gid.z;
  if (x >= P.W || y >= P.H || b >= P.B) { return; }
  let off = b * P.H * P.W;
  let yu = ((y + P.H - 1u) % P.H) * P.W;
  let yc = y * P.W;
  let yd = ((y + 1u) % P.H) * P.W;
  let xl = (x + P.W - 1u) % P.W;
  let xr = (x + 1u) % P.W;
  var idx = 0u;
  idx = idx + src[off + yu + xl] * 64u;
  idx = idx + src[off + yu + x] * 128u;
  idx = idx + src[off + yu + xr] * 256u;
  idx = idx + src[off + yc + xl] * 8u;
  idx = idx + src[off + yc + x] * 16u;
  idx = idx + src[off + yc + xr] * 32u;
  idx = idx + src[off + yd + xl] * 1u;
  idx = idx + src[off + yd + x] * 2u;
  idx = idx + src[off + yd + xr] * 4u;
  dst[off + yc + x] = rule[b * 512u + idx];
}
