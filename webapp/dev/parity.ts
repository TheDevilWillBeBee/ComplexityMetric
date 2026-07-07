/**
 * Manual browser harness: runs every system on CPU and WebGPU with
 * identical params/x0 and diffs the states. Open /dev/gpu-parity.html
 * under `npm run dev` in a WebGPU-capable browser.
 */
import { createDefaultConfig } from '../src/config'
import { SYSTEMS } from '../src/systems/registry'
import { getGpuDevice } from '../src/systems/gpu/device'
import { nullProgress } from '../src/core/progress'
import { runRollout } from '../src/systems/rollout'

interface Row {
  system: string
  steps: number
  status: 'bit-equal' | 'ok' | 'FAIL'
  maxDiff: number
  cpuMs: number
  gpuMs: number
}

const CONTINUOUS = new Set(['Coupled logistic map 1D', 'Gray-Scott 2D'])

async function runOne(name: string, device: GPUDevice): Promise<Row> {
  const spec = SYSTEMS[name]
  const cfg = createDefaultConfig()
  cfg.systemName = name
  cfg.batchSize = 2
  cfg.H = spec.defaultSize.H !== null ? 32 : null
  cfg.W = spec.spatialDim === 2 ? 32 : 64
  Object.assign(cfg, spec.paramDefaults)
  cfg.seedMode = spec.seedModes[0]
  const continuous = CONTINUOUS.has(name)
  const steps = continuous ? 10 : 200
  const opts = { steps, every: 1, skip: 0 }

  const t0 = performance.now()
  const cpu = spec.build(cfg, null)
  const cpuRoll = await runRollout(cpu.stepper, cpu.dims, cpu.meta, opts, nullProgress)
  cpu.stepper.dispose()
  const t1 = performance.now()
  const gpu = spec.build(cfg, device)
  const gpuRoll = await runRollout(gpu.stepper, gpu.dims, gpu.meta, opts, nullProgress)
  gpu.stepper.dispose()
  const t2 = performance.now()

  let maxDiff = 0
  for (let i = 0; i < cpuRoll.data.length; i++) {
    maxDiff = Math.max(maxDiff, Math.abs(cpuRoll.data[i] - gpuRoll.data[i]))
  }
  const tol = continuous ? 1e-5 : 0
  return {
    system: name,
    steps,
    status: maxDiff === 0 ? 'bit-equal' : maxDiff <= tol ? 'ok' : 'FAIL',
    maxDiff,
    cpuMs: t1 - t0,
    gpuMs: t2 - t1,
  }
}

async function benchmark(device: GPUDevice): Promise<string> {
  const cfg = createDefaultConfig()
  cfg.systemName = 'Langton CA 2D'
  cfg.batchSize = 1
  cfg.H = 256
  cfg.W = 256
  cfg.numStates = 4
  cfg.kernelSize = 3
  const spec = SYSTEMS['Langton CA 2D']
  const opts = { steps: 1000, every: 1000, skip: 0 }

  const g0 = performance.now()
  const gpu = spec.build(cfg, device)
  await runRollout(gpu.stepper, gpu.dims, gpu.meta, opts, nullProgress)
  gpu.stepper.dispose()
  const g1 = performance.now()

  const c0 = performance.now()
  const cpu = spec.build(cfg, null)
  await runRollout(cpu.stepper, cpu.dims, cpu.meta, opts, nullProgress)
  cpu.stepper.dispose()
  const c1 = performance.now()

  const speedup = (c1 - c0) / (g1 - g0)
  return `Langton 2D 256x256, 1000 steps: cpu ${(c1 - c0).toFixed(0)}ms, gpu ${(g1 - g0).toFixed(0)}ms (${speedup.toFixed(1)}x)`
}

async function main() {
  const out = document.getElementById('out')!
  const device = await getGpuDevice()
  if (!device) {
    out.textContent = 'WebGPU unavailable in this browser — nothing to compare.'
    return
  }
  const rows: Row[] = []
  for (const name of Object.keys(SYSTEMS)) {
    try {
      rows.push(await runOne(name, device))
    } catch (err) {
      rows.push({
        system: `${name} — ${err instanceof Error ? err.message : err}`,
        steps: 0,
        status: 'FAIL',
        maxDiff: NaN,
        cpuMs: 0,
        gpuMs: 0,
      })
    }
    out.innerHTML = render(rows)
  }
  out.innerHTML = render(rows) + '<p>benchmarking…</p>'
  try {
    out.innerHTML = render(rows) + `<p>${await benchmark(device)}</p>`
  } catch (err) {
    out.innerHTML = render(rows) + `<p>benchmark failed: ${err}</p>`
  }
}

function render(rows: Row[]): string {
  const cells = rows
    .map(
      (r) =>
        `<tr><td>${r.system}</td><td>${r.steps}</td>` +
        `<td class="${r.status === 'FAIL' ? 'fail' : 'pass'}">${r.status}</td>` +
        `<td>${r.maxDiff}</td><td>${r.cpuMs.toFixed(0)}</td><td>${r.gpuMs.toFixed(0)}</td></tr>`,
    )
    .join('')
  return `<table><tr><th>system</th><th>steps</th><th>status</th><th>max diff</th><th>cpu ms</th><th>gpu ms</th></tr>${cells}</table>`
}

main()
