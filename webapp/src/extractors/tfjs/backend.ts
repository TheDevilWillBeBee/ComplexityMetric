import * as tf from '@tensorflow/tfjs-core'
import '@tensorflow/tfjs-backend-cpu'

let ready: Promise<string> | null = null

/**
 * Initialize the best available tfjs backend: webgpu -> webgl -> cpu.
 * Returns the active backend name (also shown as a badge in the UI).
 */
export function initTf(): Promise<string> {
  if (!ready) {
    ready = (async () => {
      const isBrowser = typeof document !== 'undefined'
      if (isBrowser && typeof navigator !== 'undefined' && 'gpu' in navigator) {
        try {
          await import('@tensorflow/tfjs-backend-webgpu')
          if (await tf.setBackend('webgpu')) {
            await tf.ready()
            return tf.getBackend()
          }
        } catch {
          // fall through
        }
      }
      if (isBrowser) {
        try {
          await import('@tensorflow/tfjs-backend-webgl')
          if (await tf.setBackend('webgl')) {
            await tf.ready()
            return tf.getBackend()
          }
        } catch {
          // fall through
        }
      }
      await tf.setBackend('cpu')
      await tf.ready()
      return tf.getBackend()
    })()
  }
  return ready
}

/** True when the webgl backend cannot render float32 (gram precision warning). */
export function webglFloat32Capable(): boolean {
  if (tf.getBackend() !== 'webgl') return true
  try {
    return tf.env().getBool('WEBGL_RENDER_FLOAT32_CAPABLE')
  } catch {
    return true
  }
}
