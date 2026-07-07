/**
 * Core interfaces for dynamical systems and their rollouts.
 *
 * Layout convention (mirrors the Python code): a state batch is one typed
 * array in row-major (B, C, H, W) order, with H = 1 for 1D systems. A
 * rollout is (B, T, C, H, W) row-major, so the frame for (b, t) is the
 * contiguous chunk of length C*H*W at offset ((b*T + t) * C*H*W).
 */

export type StateArray = Float32Array | Int32Array

export interface Dims {
  B: number
  C: number
  /** 1 for 1D systems. */
  H: number
  W: number
}

/** Advances a simulation; one instance per run. */
export interface Stepper {
  /** Advance `count` steps. */
  step(count: number): void | Promise<void>
  /** Current state, (B, C, H, W) row-major. May return an internal buffer — copy before mutating. */
  readState(): Promise<StateArray>
  dispose(): void
}

/**
 * Writes one batch element's frame as RGBA.
 * `frame` is the (C, H, W) chunk for a single (b, t); `out` has length H*W*4.
 */
export type ToRgbFn = (
  frame: StateArray,
  C: number,
  H: number,
  W: number,
  out: Uint8ClampedArray,
) => void

export interface RolloutData {
  /** (B, T, C, H, W) row-major. */
  data: StateArray
  B: number
  T: number
  C: number
  H: number
  W: number
  spatialDim: 1 | 2
  isDiscrete: boolean
  /** Number of discrete states K, or null for continuous systems. */
  numStates: number | null
  steps: number
  every: number
  skip: number
  systemName: string
  toRgb: ToRgbFn
}

/** View of the (C,H,W) frame for batch element b at recorded index t. */
export function frameView(r: RolloutData, b: number, t: number): StateArray {
  const fsize = r.C * r.H * r.W
  const off = (b * r.T + t) * fsize
  return r.data.subarray(off, off + fsize) as StateArray
}
