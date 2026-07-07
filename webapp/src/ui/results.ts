import type { ExtractorResult, RunResult } from '../pipeline'
import type { MetricRow } from '../metrics/windows'
import { clear, el, metaBlock, warningBox } from './dom'
import { mountPlayback } from './playback'
import { drawScaled, render1dSpaceTime, render2dFrame } from './render'
import { lineChart, trajectoryPlot } from './charts'
import { tabs } from './tabs'

const fmt = (v: number | null): string =>
  v == null || Number.isNaN(v) ? '—' : Math.abs(v) < 1e-3 && v !== 0 ? v.toExponential(3) : v.toFixed(4)

function metricsTable(rows: MetricRow[]): HTMLElement {
  const table = el('table')
  const header = el('tr')
  for (const h of ['metric', 'expects', 'chunks', 'chunk size', 'stride', 'mean', 'std', 'chunk means', 'error']) {
    header.append(el('th', {}, h))
  }
  table.append(header)
  for (const row of rows) {
    const tr = el('tr')
    tr.append(
      el('td', {}, row.metric),
      el('td', {}, row.expects),
      el('td', {}, String(row.chunks)),
      el('td', {}, String(row.chunk_size)),
      el('td', {}, String(row.stride)),
      el('td', {}, fmt(row.mean)),
      el('td', {}, fmt(row.std)),
      el('td', {}, row.chunk_means),
      el('td', {}, row.error),
    )
    table.append(tr)
  }
  return table
}

function extractorPanel(result: ExtractorResult): HTMLElement {
  const panel = el('div')
  if (result.error) {
    panel.append(warningBox(`Extractor failed: ${result.error}`))
    return panel
  }
  const e = result.embedding!
  panel.append(
    metaBlock({
      input_type: result.inputType,
      output_type: result.outputType,
      embedding_shape: `(${e.B}, ${e.T}, ${e.D})`,
      embedding_discrete: e.isDiscrete,
      num_states: e.numStates ?? '—',
    }),
  )

  if (result.projectionError) {
    panel.append(warningBox(`Embedding visualization failed: ${result.projectionError}`))
  } else if (result.projections) {
    const grid = el('div', { className: 'traj-grid' })
    result.projections.forEach((points, b) => {
      grid.append(trajectoryPlot(points, e.T, 260, `batch ${b}`))
    })
    panel.append(el('h3', {}, 'Trajectory'), grid)
  }

  if (result.metricRows && result.metricRows.length > 0) {
    panel.append(el('h3', {}, 'Metrics'), metricsTable(result.metricRows))
    const curves = result.metricCurves ?? []
    const byMetric = new Map<string, { xs: number[]; ys: number[] }>()
    for (const c of curves) {
      if (!byMetric.has(c.metric)) byMetric.set(c.metric, { xs: [], ys: [] })
      const s = byMetric.get(c.metric)!
      s.xs.push(c.time)
      s.ys.push(c.mean)
    }
    // only chart when slicing produced multiple windows
    if ([...byMetric.values()].some((s) => s.xs.length > 1)) {
      panel.append(
        el('h3', {}, 'Metric slices'),
        lineChart([...byMetric.entries()].map(([label, s]) => ({ label, xs: s.xs, ys: s.ys }))),
      )
    }
  }
  return panel
}

export function renderResults(container: HTMLElement, result: RunResult): void {
  clear(container)
  const r = result.rollout

  container.append(
    el('h2', {}, 'Rollout'),
    metaBlock({
      shape: `(${r.B}, ${r.T}, ${r.C}, ${r.spatialDim === 2 ? `${r.H}, ` : ''}${r.W})`,
      discrete: r.isDiscrete,
      num_states: r.numStates ?? '—',
      spatial_dim: r.spatialDim,
      steps: r.steps,
      every: r.every,
      skip: r.skip,
      system: r.systemName,
    }),
  )

  const canvas = el('canvas')
  container.append(canvas)
  if (r.spatialDim === 1) {
    drawScaled(canvas, render1dSpaceTime(r), 1200)
  } else {
    drawScaled(canvas, render2dFrame(r), 1200)
    if (r.T > 1) {
      const box = el('div', { className: 'playback' }, el('h3', {}, 'Playback'))
      container.append(box)
      mountPlayback(box, r)
    }
  }

  if (result.extractorResults.length > 0) {
    container.append(el('h2', {}, 'Feature Extractors'))
    container.append(
      tabs(
        result.extractorResults.map((res) => ({
          label: res.name,
          panel: extractorPanel(res),
        })),
      ),
    )
  }
}
