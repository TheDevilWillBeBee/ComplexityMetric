import * as tf from '@tensorflow/tfjs-core'

/**
 * Circular (wrap) padding along H and W of an NHWC tensor — tf.pad has no
 * wrap mode, so concatenate boundary slices instead. Kernels here are
 * always odd, so padding is symmetric.
 */
export function circularPad(x: tf.Tensor4D, ph: number, pw: number): tf.Tensor4D {
  let y = x
  if (pw > 0) {
    const w = y.shape[2]
    y = tf.concat(
      [tf.slice(y, [0, 0, w - pw, 0], [-1, -1, pw, -1]), y, tf.slice(y, [0, 0, 0, 0], [-1, -1, pw, -1])],
      2,
    )
  }
  if (ph > 0) {
    const h = y.shape[1]
    y = tf.concat(
      [tf.slice(y, [0, h - ph, 0, 0], [-1, ph, -1, -1]), y, tf.slice(y, [0, 0, 0, 0], [-1, ph, -1, -1])],
      1,
    )
  }
  return y
}

/**
 * Instance norm with identity affine (PyTorch InstanceNorm at init:
 * affine weight=1 bias=0, eps=1e-5, no running stats): per-instance,
 * per-channel normalization over the spatial axes.
 */
export function instanceNorm(x: tf.Tensor4D, eps = 1e-5): tf.Tensor4D {
  const { mean, variance } = tf.moments(x, [1, 2], true)
  return tf.div(tf.sub(x, mean), tf.sqrt(tf.add(variance, eps))) as tf.Tensor4D
}

/**
 * Gram reduction (extractors.py _reduce_spatial): (n, H, W, c) ->
 * (n, c*c) via f f^T / spatial_size on the channel vectors.
 */
export function gramReduce(x: tf.Tensor4D): tf.Tensor2D {
  const [n, h, w, c] = x.shape
  const f = tf.reshape(x, [n, h * w, c])
  const gram = tf.div(tf.matMul(f, f, true, false), h * w)
  return tf.reshape(gram, [n, c * c]) as tf.Tensor2D
}

/** Row-wise L2 normalization (F.normalize(dim=-1), eps 1e-12). */
export function l2Normalize(x: tf.Tensor2D): tf.Tensor2D {
  const norm = tf.maximum(tf.norm(x, 'euclidean', 1, true), 1e-12)
  return tf.div(x, norm) as tf.Tensor2D
}
