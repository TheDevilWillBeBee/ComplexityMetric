import type { AppConfig } from './config'
import type { ProgressReporter } from './core/progress'
import { nullProgress } from './core/progress'
import type { RolloutData } from './systems/types'
import { SYSTEMS } from './systems/registry'
import { getGpuDevice } from './systems/gpu/device'
import { runRollout } from './systems/rollout'
import type { EmbeddingData } from './extractors/types'
import { embeddingBatchView } from './extractors/types'
import { EXTRACTORS } from './extractors/registry'
import { project } from './projection'
import type { CurvePoint, MetricRow } from './metrics/windows'
import { computeMetrics } from './metrics/windows'

export interface ExtractorResult {
  name: string
  inputType: string
  outputType: string
  embedding?: EmbeddingData
  /** Per batch element, (T, 2) row-major. */
  projections?: Float32Array[]
  projectionError?: string
  metricRows?: MetricRow[]
  metricCurves?: CurvePoint[]
  error?: string
}

export interface RunResult {
  rollout: RolloutData
  extractorResults: ExtractorResult[]
  systemsBackend: 'webgpu' | 'cpu'
}

function selectedExtractors(cfg: AppConfig): string[] {
  const spec = SYSTEMS[cfg.systemName]
  return cfg.extractors.filter((name) => {
    const e = EXTRACTORS[name]
    return e && e.dims.includes(spec.spatialDim)
  })
}

/** Stage list with weights, used to configure the progress bar. */
export function planStages(cfg: AppConfig): { label: string; weight: number }[] {
  const names = selectedExtractors(cfg)
  const hasMetrics = cfg.metrics.length > 0
  const stages = [{ label: 'Simulating', weight: 0.25 }]
  for (const name of names) stages.push({ label: `Extracting ${name}`, weight: 0.35 / Math.max(1, names.length) })
  stages.push({ label: 'Projecting', weight: 0.1 })
  if (hasMetrics) stages.push({ label: 'Computing metrics', weight: 0.3 })
  return stages
}

export async function runPipeline(
  cfg: AppConfig,
  progress: ProgressReporter = nullProgress,
): Promise<RunResult> {
  const spec = SYSTEMS[cfg.systemName]
  if (!spec) throw new Error(`unknown system: ${cfg.systemName}`)

  progress.stage('Simulating')
  const device = await getGpuDevice()
  const built = spec.build(cfg, device)
  let rollout: RolloutData
  try {
    rollout = await runRollout(
      built.stepper,
      built.dims,
      built.meta,
      { steps: cfg.steps, every: cfg.every, skip: cfg.skip },
      progress,
    )
  } finally {
    built.stepper.dispose()
  }

  const extractorResults: ExtractorResult[] = []
  for (const name of selectedExtractors(cfg)) {
    const extSpec = EXTRACTORS[name]
    progress.stage(`Extracting ${name}`)
    const result: ExtractorResult = {
      name,
      inputType: extSpec.inputType,
      outputType: extSpec.outputType,
    }
    extractorResults.push(result)
    const extractor = await extSpec.make(cfg)
    try {
      result.embedding = await extractor.extract(rollout, progress)
    } catch (err) {
      result.error = err instanceof Error ? err.message : String(err)
    } finally {
      extractor.dispose()
    }
  }

  progress.stage('Projecting')
  let done = 0
  const total = extractorResults.filter((r) => r.embedding).length || 1
  for (const result of extractorResults) {
    if (!result.embedding) continue
    try {
      const e = result.embedding
      const projections: Float32Array[] = []
      for (let b = 0; b < e.B; b++) {
        const z =
          embeddingBatchView(e, b) instanceof Float32Array
            ? (embeddingBatchView(e, b) as Float32Array)
            : Float32Array.from(embeddingBatchView(e, b))
        projections.push(await project(cfg.dimMethod, z, e.T, e.D))
        await progress.tick()
      }
      result.projections = projections
    } catch (err) {
      result.projectionError = err instanceof Error ? err.message : String(err)
    }
    done++
    progress.set(done / total)
  }

  if (cfg.metrics.length > 0) {
    progress.stage('Computing metrics')
    const withEmbedding = extractorResults.filter((r) => r.embedding)
    let mDone = 0
    for (const result of withEmbedding) {
      const chunkSize = cfg.metricChunksEnabled ? cfg.metricChunkSize : 0
      const stride = cfg.metricChunksEnabled ? cfg.metricStride : 1
      const { rows, curves } = await computeMetrics(
        result.embedding!,
        cfg.metrics,
        { chunkSize, stride },
        progress,
      )
      result.metricRows = rows
      result.metricCurves = curves
      mDone++
      progress.set(mDone / (withEmbedding.length || 1))
      await progress.tick()
    }
  }

  return { rollout, extractorResults, systemsBackend: device ? 'webgpu' : 'cpu' }
}
