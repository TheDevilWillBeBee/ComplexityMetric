let devicePromise: Promise<GPUDevice | null> | null = null

/** Singleton WebGPU device; resolves to null when WebGPU is unavailable. */
export function getGpuDevice(): Promise<GPUDevice | null> {
  if (!devicePromise) {
    devicePromise = (async () => {
      try {
        if (typeof navigator === 'undefined' || !('gpu' in navigator)) return null
        const adapter = await navigator.gpu.requestAdapter()
        if (!adapter) return null
        const device = await adapter.requestDevice()
        device.lost.then(() => {
          devicePromise = null
        })
        return device
      } catch {
        return null
      }
    })()
  }
  return devicePromise
}
