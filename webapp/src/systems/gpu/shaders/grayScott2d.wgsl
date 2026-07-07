// GrayScott2D: forward Euler with the 5-point Laplacian, dt = f0.
// State layout: u at (b*2)*H*W, v at (b*2+1)*H*W.
// sp layout per batch: sp[b*4 + 0..3] = Du, Dv, F, k.
struct SimParams {
  B: u32, C: u32, H: u32, W: u32,
  K: u32, numStates: u32, tableLen: u32, flags: u32,
  f0: f32, f1: f32, f2: f32, f3: f32,
};

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> sp: array<f32>; // (B, 4)

@compute @workgroup_size(8, 8, 1)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let y = gid.y;
  let b = gid.z;
  if (x >= P.W || y >= P.H || b >= P.B) { return; }
  let hw = P.H * P.W;
  let uOff = b * 2u * hw;
  let vOff = uOff + hw;
  let yu = ((y + P.H - 1u) % P.H) * P.W;
  let yc = y * P.W;
  let yd = ((y + 1u) % P.H) * P.W;
  let xl = (x + P.W - 1u) % P.W;
  let xr = (x + 1u) % P.W;

  let u = src[uOff + yc + x];
  let v = src[vOff + yc + x];
  let lapU = src[uOff + yu + x] + src[uOff + yd + x] + src[uOff + yc + xl] + src[uOff + yc + xr] - 4.0 * u;
  let lapV = src[vOff + yu + x] + src[vOff + yd + x] + src[vOff + yc + xl] + src[vOff + yc + xr] - 4.0 * v;

  let Du = sp[b * 4u];
  let Dv = sp[b * 4u + 1u];
  let F = sp[b * 4u + 2u];
  let k = sp[b * 4u + 3u];
  let uvv = u * v * v;
  let dt = P.f0;
  dst[uOff + yc + x] = clamp(u + dt * (Du * lapU - uvv + F * (1.0 - u)), 0.0, 1.0);
  dst[vOff + yc + x] = clamp(v + dt * (Dv * lapV + uvv - (F + k) * v), 0.0, 1.0);
}
