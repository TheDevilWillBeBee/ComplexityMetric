import { el } from './dom'

/** Hand-rolled tab strip: returns the container; panels are lazily shown. */
export function tabs(items: { label: string; panel: HTMLElement }[]): HTMLElement {
  const strip = el('div', { className: 'tabs' })
  const panels = el('div')
  const buttons: HTMLButtonElement[] = []

  items.forEach((item, i) => {
    const btn = el('button', { className: 'tab-btn' }, item.label) as HTMLButtonElement
    item.panel.classList.add('tab-panel')
    if (i !== 0) item.panel.classList.add('hidden')
    else btn.classList.add('active')
    btn.addEventListener('click', () => {
      buttons.forEach((b, j) => {
        b.classList.toggle('active', j === i)
        items[j].panel.classList.toggle('hidden', j !== i)
      })
    })
    buttons.push(btn)
    strip.append(btn)
    panels.append(item.panel)
  })

  return el('div', {}, strip, panels)
}
