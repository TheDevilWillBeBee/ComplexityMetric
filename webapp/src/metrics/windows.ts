import type { EmbeddingData } from '../extractors/types'
import { sliceEmbedding } from '../extractors/types'
import type { ProgressReporter } from '../core/progress'
import { nullProgress } from '../core/progress'
import { METRICS } from './registry'
import { embeddingToContinuous, embeddingToDiscrete } from './convert'

/** Port of metric_windows (app_dynamics.py:378-383). */
export function metricWindows(T: number, chunkSize: number, stride: number): Array<[number, number]> {
  if (T <= 0) return []
  const size = chunkSize <= 0 ? T : Math.min(Math.floor(chunkSize), T)
  const step = Math.max(1, Math.floor(stride))
  const out: Array<[number, number]> = []
  for (let start = 0; start + size <= T; start += step) out.push([start, start + size])
  return out
}

export interface MetricRow {
  metric: string
  expects: string
  chunks: number
  chunk_size: number
  stride: number
  mean: number | null
  std: number | null
  chunk_means: string
  error: string
}

export interface CurvePoint {
  metric: string
  chunk: number
  start: number
  end: number
  time: number
  mean: number
  std: number
}

function sampleStats(values: number[]): { mean: number; std: number } {
  const n = values.length
  if (n === 0) return { mean: NaN, std: 0 }
  let mean = 0
  for (const v of values) mean += v
  mean /= n
  if (n === 1) return { mean, std: 0 }
  let varSum = 0
  for (const v of values) varSum += (v - mean) ** 2
  // torch .std() default: unbiased
  return { mean, std: Math.sqrt(varSum / (n - 1)) }
}

/**
 * Port of compute_metrics (app_dynamics.py:386-448): score each selected
 * metric over the given windows; per-metric errors land in the table's
 * error column instead of failing the run.
 */
export async function computeMetrics(
  embedding: EmbeddingData,
  names: string[],
  opts: { chunkSize: number; stride: number },
  progress: ProgressReporter = nullProgress,
): Promise<{ rows: MetricRow[]; curves: CurvePoint[] }> {
  const windows = metricWindows(embedding.T, opts.chunkSize, opts.stride)
  const rows: MetricRow[] = []
  const curves: CurvePoint[] = []

  for (let mi = 0; mi < names.length; mi++) {
    const name = names[mi]
    const spec = METRICS[name]
    const base: Omit<MetricRow, 'mean' | 'std' | 'chunk_means' | 'error'> = {
      metric: name,
      expects: spec?.inputType ?? '?',
      chunks: windows.length,
      chunk_size: windows.length === 0 ? 0 : windows[0][1] - windows[0][0],
      stride: Math.max(1, Math.floor(opts.stride)),
    }
    try {
      if (!spec) throw new Error(`unknown metric: ${name}`)
      const metric = spec.make()
      const allScores: number[] = []
      const chunkMeans: number[] = []
      const metricCurves: CurvePoint[] = []
      for (let ci = 0; ci < windows.length; ci++) {
        const [start, end] = windows[ci]
        const window = sliceEmbedding(embedding, start, end)
        const typed =
          spec.inputType === 'discrete' ? embeddingToDiscrete(window) : embeddingToContinuous(window)
        const scores = Array.from(await metric.compute(typed))
        allScores.push(...scores)
        const { mean, std } = sampleStats(scores)
        chunkMeans.push(mean)
        metricCurves.push({
          metric: name,
          chunk: ci,
          start,
          end,
          time: start + (end - start - 1) / 2,
          mean,
          std,
        })
        await progress.tick()
      }
      const { mean, std } = sampleStats(allScores)
      curves.push(...metricCurves)
      rows.push({
        ...base,
        mean,
        std,
        chunk_means: chunkMeans.map((v) => v.toFixed(4)).join(', '),
        error: '',
      })
    } catch (err) {
      rows.push({
        ...base,
        mean: null,
        std: null,
        chunk_means: '',
        error: err instanceof Error ? err.message : String(err),
      })
    }
    progress.set((mi + 1) / names.length)
  }
  return { rows, curves }
}
