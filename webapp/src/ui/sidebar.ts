import type { AppConfig } from '../config'
import { SYSTEMS, type ParamControl } from '../systems/registry'
import { availableExtractors, defaultExtractors } from '../extractors/registry'
import { METRICS, METRIC_NAMES } from '../metrics/registry'
import { PROJECTION_METHODS } from '../projection'
import { checkbox, multiSelect, numberInput, section, select, slider, textInput } from './controls'
import { clear, el } from './dom'

function metricTypesInfo(): HTMLElement {
  const details = el('details', { className: 'info-table' }, el('summary', {}, 'Metric input types'))
  const table = el('table')
  for (const [name, spec] of Object.entries(METRICS)) {
    table.append(el('tr', {}, el('td', {}, name), el('td', {}, spec.inputType)))
  }
  details.append(table)
  return details
}

/**
 * Sidebar mirroring app_dynamics.py sidebar(). Controls write straight into
 * `cfg`; selecting a system applies that system's defaults and re-renders.
 */
export function mountSidebar(root: HTMLElement, cfg: AppConfig): void {
  const render = () => {
    clear(root)
    const spec = SYSTEMS[cfg.systemName]

    const paramControl = (c: ParamControl): HTMLElement => {
      switch (c.kind) {
        case 'number':
          return numberInput(
            c.label,
            () => cfg[c.key] as number,
            (v) => ((cfg[c.key] as number) = v),
            { min: c.min, max: c.max, step: c.step },
          )
        case 'slider':
          return slider(
            c.label,
            () => cfg[c.key] as number,
            (v) => ((cfg[c.key] as number) = v),
            { min: c.min, max: c.max, step: c.step },
          )
        case 'checkbox':
          return checkbox(
            c.label,
            () => cfg[c.key] as boolean,
            (v) => ((cfg[c.key] as boolean) = v),
          )
        case 'text':
          return textInput(
            c.label,
            () => cfg[c.key] as string,
            (v) => ((cfg[c.key] as string) = v),
          )
      }
    }

    root.append(
      el('h2', {}, 'Dynamical Systems'),
      section(
        'Simulation',
        select('System', Object.keys(SYSTEMS), () => cfg.systemName, (name) => {
          const next = SYSTEMS[name]
          cfg.systemName = name
          cfg.H = next.defaultSize.H
          cfg.W = next.defaultSize.W
          cfg.steps = next.defaultSteps.steps
          cfg.every = next.defaultSteps.every
          cfg.skip = next.defaultSteps.skip
          Object.assign(cfg, next.paramDefaults)
          if (!next.seedModes.includes(cfg.seedMode)) cfg.seedMode = next.seedModes[0]
          cfg.extractors = defaultExtractors(next.isDiscrete).filter((e) =>
            availableExtractors(next.spatialDim).includes(e),
          )
          render()
        }),
        numberInput('Batch', () => cfg.batchSize, (v) => (cfg.batchSize = v), { min: 1, max: 16 }),
        spec.spatialDim === 2
          ? numberInput('Height', () => cfg.H ?? 64, (v) => (cfg.H = v), { min: 16, max: 256, step: 8 })
          : null,
        numberInput('Width', () => cfg.W, (v) => (cfg.W = v), {
          min: 16,
          max: spec.spatialDim === 2 ? 256 : 1024,
          step: spec.spatialDim === 2 ? 8 : 16,
        }),
        numberInput('Steps', () => cfg.steps, (v) => (cfg.steps = v), { min: 1, max: 10000 }),
        numberInput('Every', () => cfg.every, (v) => (cfg.every = v), { min: 1, max: 200 }),
        numberInput('Skip', () => cfg.skip, (v) => (cfg.skip = v), { min: 0, max: 10000 }),
      ),
      section('System Parameters', ...spec.paramControls.map(paramControl)),
      section(
        'Initial State',
        select('Seed mode', spec.seedModes, () => cfg.seedMode, (v) => (cfg.seedMode = v as AppConfig['seedMode'])),
        spec.isDiscrete
          ? slider('Initial density', () => cfg.seedP, (v) => (cfg.seedP = v), { min: 0, max: 1, step: 0.01 })
          : null,
      ),
      section(
        'Feature Extractors',
        multiSelect(
          'Extractors',
          availableExtractors(spec.spatialDim),
          () => cfg.extractors,
          (v) => (cfg.extractors = v),
        ),
      ),
      section(
        'Embedding',
        select(
          'Trajectory projection',
          PROJECTION_METHODS,
          () => cfg.dimMethod,
          (v) => (cfg.dimMethod = v as AppConfig['dimMethod']),
        ),
        numberInput('Feature dimension', () => cfg.embedDim, (v) => (cfg.embedDim = v), {
          min: 2,
          max: 512,
          step: 8,
        }),
        numberInput('Extractor seed', () => cfg.extractorSeed, (v) => (cfg.extractorSeed = v), {
          min: 0,
          max: 2147483647,
        }),
      ),
      section(
        'Metrics',
        multiSelect('Metrics', METRIC_NAMES, () => cfg.metrics, (v) => (cfg.metrics = v)),
        metricTypesInfo(),
        checkbox(
          'Temporal metric slices',
          () => cfg.metricChunksEnabled,
          (v) => (cfg.metricChunksEnabled = v),
        ),
        numberInput('Metric chunk size', () => cfg.metricChunkSize, (v) => (cfg.metricChunkSize = v), {
          min: 0,
          max: 100000,
        }),
        numberInput('Metric stride', () => cfg.metricStride, (v) => (cfg.metricStride = v), {
          min: 1,
          max: 100000,
        }),
      ),
      section(
        'Randomness',
        numberInput('App seed', () => cfg.appSeed, (v) => (cfg.appSeed = v), { min: 0, max: 2147483647 }),
      ),
    )
  }
  render()
}
