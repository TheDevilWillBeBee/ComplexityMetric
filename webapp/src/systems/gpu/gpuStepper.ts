import type { Dims, StateArray, Stepper } from '../types'
import type { ProgressReporter } from '../../core/progress'
import type { RolloutOptions } from '../rollout'
import { isRecorded } from '../rollout'

/**
 * Generic WebGPU stepper: one compute pass per simulation step over
 * ping-pong state buffers, many passes per command submit. Frame recording
 * (`rolloutInto`) copies recorded states into a stash buffer on the GPU
 * and drains it through a staging buffer with mapAsync — far fewer sync
 * points than a readback per frame.
 *
 * Bind layout: 0 = SimParams uniform, 1 = src state, 2 = dst state,
 * 3.. = read-only storage buffers (rule tables / per-batch params).
 */
export interface GpuSystemDesc {
  /** WGSL with an entry point `step` and the bind layout above. */
  code: string
  label: string
  dims: Dims
  /** Uniform contents: 8 u32 followed by 4 f32 (48 bytes). */
  uniformInts: [number, number, number, number, number, number, number, number]
  uniformFloats: [number, number, number, number]
  storages: Array<Uint32Array | Int32Array | Float32Array>
  workgroup: '1d' | '2d'
}

/** A Stepper with a fast full-rollout path used by runRollout when present. */
export interface GpuStepper extends Stepper {
  rolloutInto(
    data: StateArray,
    T: number,
    opts: RolloutOptions,
    progress: ProgressReporter,
  ): Promise<number> // number of frames recorded
}

const STASH_BYTE_CAP = 32 * 1024 * 1024
const STEPS_PER_SUBMIT = 128

export function makeGpuStepper(
  device: GPUDevice,
  desc: GpuSystemDesc,
  x0: StateArray,
): GpuStepper {
  const { B, C, H, W } = desc.dims
  const stateLen = B * C * H * W
  const stateBytes = stateLen * 4
  const discrete = !(x0 instanceof Float32Array)

  const module = device.createShaderModule({ code: desc.code, label: desc.label })
  const pipeline = device.createComputePipeline({
    label: desc.label,
    layout: 'auto',
    compute: { module, entryPoint: 'step' },
  })

  // uniform
  const uniformData = new ArrayBuffer(48)
  new Uint32Array(uniformData, 0, 8).set(desc.uniformInts)
  new Float32Array(uniformData, 32, 4).set(desc.uniformFloats)
  const uniform = device.createBuffer({
    size: 48,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    label: `${desc.label}-uniform`,
  })
  device.queue.writeBuffer(uniform, 0, uniformData)

  // state ping-pong
  const mkState = (label: string) =>
    device.createBuffer({
      size: stateBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label,
    })
  const stateA = mkState(`${desc.label}-stateA`)
  const stateB = mkState(`${desc.label}-stateB`)
  device.queue.writeBuffer(stateA, 0, x0.buffer, x0.byteOffset, stateBytes)

  // extra storages
  const storages = desc.storages.map((arr, i) => {
    const buf = device.createBuffer({
      size: Math.max(16, arr.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: `${desc.label}-storage${i}`,
    })
    device.queue.writeBuffer(buf, 0, arr.buffer, arr.byteOffset, arr.byteLength)
    return buf
  })

  const mkBindGroup = (src: GPUBuffer, dst: GPUBuffer) =>
    device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uniform } },
        { binding: 1, resource: { buffer: src } },
        { binding: 2, resource: { buffer: dst } },
        ...storages.map((buf, i) => ({ binding: 3 + i, resource: { buffer: buf } })),
      ],
    })
  const bindGroups = [mkBindGroup(stateA, stateB), mkBindGroup(stateB, stateA)]
  const states = [stateA, stateB]
  let cur = 0 // index of buffer holding the CURRENT state

  const encodeStep = (encoder: GPUCommandEncoder) => {
    const pass = encoder.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bindGroups[cur])
    if (desc.workgroup === '2d') {
      pass.dispatchWorkgroups(Math.ceil(W / 8), Math.ceil(H / 8), B)
    } else {
      pass.dispatchWorkgroups(Math.ceil(W / 256), B, 1)
    }
    pass.end()
    cur ^= 1
  }

  const staging = device.createBuffer({
    size: stateBytes,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    label: `${desc.label}-staging`,
  })

  const readCurrent = async (): Promise<StateArray> => {
    const encoder = device.createCommandEncoder()
    encoder.copyBufferToBuffer(states[cur], 0, staging, 0, stateBytes)
    device.queue.submit([encoder.finish()])
    await staging.mapAsync(GPUMapMode.READ)
    const out: StateArray = discrete ? new Int32Array(stateLen) : new Float32Array(stateLen)
    out.set(
      discrete
        ? new Int32Array(staging.getMappedRange(0, stateBytes))
        : new Float32Array(staging.getMappedRange(0, stateBytes)),
    )
    staging.unmap()
    return out
  }

  return {
    step(count: number) {
      let remaining = count
      while (remaining > 0) {
        const n = Math.min(remaining, STEPS_PER_SUBMIT)
        const encoder = device.createCommandEncoder()
        for (let i = 0; i < n; i++) encodeStep(encoder)
        device.queue.submit([encoder.finish()])
        remaining -= n
      }
    },

    readState: readCurrent,

    async rolloutInto(data, T, opts, progress) {
      const frameSize = C * H * W
      const stashFrames = Math.max(1, Math.min(T, Math.floor(STASH_BYTE_CAP / stateBytes)))
      const stash = device.createBuffer({
        size: stashFrames * stateBytes,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        label: `${desc.label}-stash`,
      })
      const stashStaging = device.createBuffer({
        size: stashFrames * stateBytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        label: `${desc.label}-stash-staging`,
      })

      let slot = 0
      let tRec = 0
      const pending: number[] = [] // recorded-frame indices currently in the stash

      const drain = async () => {
        if (pending.length === 0) return
        const bytes = pending.length * stateBytes
        const encoder = device.createCommandEncoder()
        encoder.copyBufferToBuffer(stash, 0, stashStaging, 0, bytes)
        device.queue.submit([encoder.finish()])
        await stashStaging.mapAsync(GPUMapMode.READ, 0, bytes)
        const mapped = discrete
          ? new Int32Array(stashStaging.getMappedRange(0, bytes))
          : new Float32Array(stashStaging.getMappedRange(0, bytes))
        for (let i = 0; i < pending.length; i++) {
          const frame = pending[i]
          for (let b = 0; b < B; b++) {
            const src = mapped.subarray(i * stateLen + b * frameSize, i * stateLen + (b + 1) * frameSize)
            data.set(src as never, (b * T + frame) * frameSize)
          }
        }
        stashStaging.unmap()
        pending.length = 0
        slot = 0
      }

      let encoder: GPUCommandEncoder | null = null
      let encoded = 0
      const flush = () => {
        if (encoder) {
          device.queue.submit([encoder.finish()])
          encoder = null
          encoded = 0
        }
      }

      for (let t = 0; t < opts.steps; t++) {
        if (!encoder) encoder = device.createCommandEncoder()
        encodeStep(encoder)
        encoded++
        if (isRecorded(t, opts.every, opts.skip)) {
          encoder.copyBufferToBuffer(states[cur], 0, stash, slot * stateBytes, stateBytes)
          pending.push(tRec)
          slot++
          tRec++
          if (slot === stashFrames) {
            flush()
            await drain()
            progress.set(opts.steps > 0 ? t / opts.steps : 1)
            await progress.tick()
          }
        }
        if (encoded >= STEPS_PER_SUBMIT) {
          flush()
          progress.set(opts.steps > 0 ? t / opts.steps : 1)
          await progress.tick()
        }
      }
      flush()
      await drain()
      stash.destroy()
      stashStaging.destroy()
      return tRec
    },

    dispose() {
      stateA.destroy()
      stateB.destroy()
      staging.destroy()
      uniform.destroy()
      for (const buf of storages) buf.destroy()
    },
  }
}
