// CoupledLogistic1D: x' = clamp((1-eps)*f(x) + eps/2*(f(left)+f(right)), 0, 1)
// with f(v) = r*v*(1-v). sp layout per batch: sp[b*2] = r, sp[b*2+1] = eps.
struct SimParams {
  B: u32, C: u32, H: u32, W: u32,
  K: u32, numStates: u32, tableLen: u32, flags: u32,
  f0: f32, f1: f32, f2: f32, f3: f32,
};

@group(0) @binding(0) var<uniform> P: SimParams;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<storage, read> sp: array<f32>; // (B, 2)

fn logi(v: f32, r: f32) -> f32 {
  return r * v * (1.0 - v);
}

@compute @workgroup_size(256, 1, 1)
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x;
  let b = gid.y;
  if (x >= P.W || b >= P.B) { return; }
  let off = b * P.W;
  let r = sp[b * 2u];
  let eps = sp[b * 2u + 1u];
  let fc = logi(src[off + x], r);
  let fl = logi(src[off + (x + P.W - 1u) % P.W], r);
  let fr = logi(src[off + (x + 1u) % P.W], r);
  dst[off + x] = clamp((1.0 - eps) * fc + 0.5 * eps * (fl + fr), 0.0, 1.0);
}
