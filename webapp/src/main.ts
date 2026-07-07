import { createDefaultConfig } from './config'
import { EXTRACTORS, availableExtractors } from './extractors/registry'
import { getGpuDevice } from './systems/gpu/device'
import { planStages, runPipeline } from './pipeline'
import { SYSTEMS } from './systems/registry'
import { clear, el, warningBox } from './ui/dom'

import { mountProgressBar } from './ui/progressBar'
import { renderResults } from './ui/results'
import { mountSidebar } from './ui/sidebar'

function main() {
  const app = document.getElementById('app')!
  const sidebar = el('aside', { id: 'sidebar' })
  const note = el('div', { className: 'info' })
  const runBtn = el('button', { id: 'run', className: 'primary' }, 'Run simulation') as HTMLButtonElement
  const progressHost = el('div', { id: 'progress' })
  const results = el('div', { id: 'results' })
  const mainCol = el(
    'main',
    { id: 'main' },
    el('h1', {}, 'Dynamical Systems Explorer'),
    el('p', { className: 'caption' }, 'Standalone browser port of app_dynamics.py — everything runs locally.'),
    note,
    runBtn,
    progressHost,
    results,
  )
  app.append(sidebar, mainCol)

  const cfg = createDefaultConfig()
  const progress = mountProgressBar(progressHost)

  const badges = el('div', { className: 'badges' })
  note.after(badges)
  const sysBadge = el('span', { className: 'badge' }, 'systems: …')
  badges.append(sysBadge)
  getGpuDevice()
    .then((device) => (sysBadge.textContent = `systems: ${device ? 'webgpu' : 'cpu'}`))
    .catch(() => (sysBadge.textContent = 'systems: cpu'))
  const tfBadge = el('span', { className: 'badge' }, 'tfjs: loading…')
  badges.append(tfBadge)
  // dynamic import keeps tfjs out of the initial bundle
  import('./extractors/tfjs/backend')
    .then(async (tfjs) => {
      const backend = await tfjs.initTf()
      tfBadge.textContent = `tfjs: ${backend}`
      if (!tfjs.webglFloat32Capable()) {
        badges.append(el('span', { className: 'badge warn' }, 'webgl float16 — RandomConvNet/VGG precision reduced'))
      }
    })
    .catch(() => (tfBadge.textContent = 'tfjs: unavailable'))

  const extractorInfo = el('details', { className: 'info-table' })
  note.after(extractorInfo)

  const updateNote = () => {
    const spec = SYSTEMS[cfg.systemName]
    note.textContent = `${spec.name}: ${spec.note}`
    clear(extractorInfo)
    extractorInfo.append(el('summary', {}, 'Extractor info'))
    const table = el('table')
    const header = el('tr')
    for (const h of ['extractor', 'input', 'output', 'available', 'note']) header.append(el('th', {}, h))
    table.append(header)
    const available = new Set(availableExtractors(spec.spatialDim))
    for (const ext of Object.values(EXTRACTORS)) {
      const tr = el('tr')
      tr.append(
        el('td', {}, ext.name),
        el('td', {}, ext.inputType),
        el('td', {}, ext.outputType),
        el('td', {}, available.has(ext.name) ? 'yes' : 'no'),
        el('td', {}, ext.note),
      )
      table.append(tr)
    }
    extractorInfo.append(table)
  }
  updateNote()
  // sidebar re-renders on system change; keep the note in sync via event delegation
  sidebar.addEventListener('change', updateNote)
  mountSidebar(sidebar, cfg)

  runBtn.addEventListener('click', async () => {
    runBtn.disabled = true
    clear(results)
    progress.begin(planStages(cfg))
    try {
      const result = await runPipeline(structuredClone(cfg), progress.reporter)
      progress.finish()
      renderResults(results, result)
    } catch (err) {
      progress.hide()
      results.append(warningBox(err instanceof Error ? err.message : String(err)))
    } finally {
      runBtn.disabled = false
    }
  })
}

main()
