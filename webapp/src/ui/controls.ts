/** Sidebar form controls. Each control reads/writes a config object directly. */
import { el } from './dom'

export function section(title: string, ...children: (HTMLElement | null)[]): HTMLElement {
  const box = el('div', { className: 'section' }, el('h3', {}, title))
  for (const c of children) if (c) box.append(c)
  return box
}

function field(label: string, input: HTMLElement, suffix?: HTMLElement): HTMLElement {
  const wrap = el('label', { className: 'field' }, el('span', { className: 'field-label' }, label))
  wrap.append(input)
  if (suffix) wrap.append(suffix)
  return wrap
}

export function numberInput(
  label: string,
  get: () => number,
  set: (v: number) => void,
  opts: { min: number; max: number; step?: number },
): HTMLElement {
  const input = el('input', {
    type: 'number',
    min: opts.min,
    max: opts.max,
    step: opts.step ?? 1,
    value: get(),
  }) as HTMLInputElement
  input.addEventListener('change', () => {
    let v = Number(input.value)
    if (!Number.isFinite(v)) v = opts.min
    v = Math.min(opts.max, Math.max(opts.min, v))
    input.value = String(v)
    set(v)
  })
  return field(label, input)
}

export function slider(
  label: string,
  get: () => number,
  set: (v: number) => void,
  opts: { min: number; max: number; step: number },
): HTMLElement {
  const input = el('input', {
    type: 'range',
    min: opts.min,
    max: opts.max,
    step: opts.step,
    value: get(),
  }) as HTMLInputElement
  const value = el('span', { className: 'slider-value' }, String(get()))
  input.addEventListener('input', () => {
    value.textContent = input.value
    set(Number(input.value))
  })
  return field(label, input, value)
}

export function checkbox(label: string, get: () => boolean, set: (v: boolean) => void): HTMLElement {
  const input = el('input', { type: 'checkbox' }) as HTMLInputElement
  input.checked = get()
  input.addEventListener('change', () => set(input.checked))
  const wrap = el('label', { className: 'field field-checkbox' })
  wrap.append(input, el('span', { className: 'field-label' }, label))
  return wrap
}

export function textInput(label: string, get: () => string, set: (v: string) => void): HTMLElement {
  const input = el('input', { type: 'text', value: get() }) as HTMLInputElement
  input.addEventListener('change', () => set(input.value))
  return field(label, input)
}

export function select(
  label: string,
  options: string[],
  get: () => string,
  set: (v: string) => void,
): HTMLElement {
  const input = el('select') as HTMLSelectElement
  for (const opt of options) input.append(el('option', { value: opt }, opt))
  input.value = get()
  input.addEventListener('change', () => set(input.value))
  return field(label, input)
}

/** Streamlit-multiselect stand-in: a checkbox list. */
export function multiSelect(
  label: string,
  options: string[],
  get: () => string[],
  set: (v: string[]) => void,
): HTMLElement {
  const box = el('div', { className: 'field multi' }, el('span', { className: 'field-label' }, label))
  const selected = new Set(get())
  for (const opt of options) {
    const input = el('input', { type: 'checkbox' }) as HTMLInputElement
    input.checked = selected.has(opt)
    input.addEventListener('change', () => {
      if (input.checked) selected.add(opt)
      else selected.delete(opt)
      set(options.filter((o) => selected.has(o)))
    })
    const item = el('label', { className: 'multi-item' })
    item.append(input, el('span', {}, opt))
    box.append(item)
  }
  return box
}
