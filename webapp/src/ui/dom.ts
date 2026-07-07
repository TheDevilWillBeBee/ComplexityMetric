/** Tiny DOM construction helpers — no framework. */

type Child = Node | string | null | undefined

export function el<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | boolean | number> = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag)
  for (const [key, value] of Object.entries(attrs)) {
    if (key === 'className') node.className = String(value)
    else if (typeof value === 'boolean') {
      if (value) node.setAttribute(key, '')
    } else node.setAttribute(key, String(value))
  }
  for (const child of children) {
    if (child == null) continue
    node.append(child instanceof Node ? child : document.createTextNode(child))
  }
  return node
}

export function clear(node: HTMLElement): void {
  while (node.firstChild) node.removeChild(node.firstChild)
}

/** Simple key/value metadata block (stands in for st.write(dict)). */
export function metaBlock(entries: Record<string, unknown>): HTMLElement {
  const dl = el('dl', { className: 'meta' })
  for (const [key, value] of Object.entries(entries)) {
    dl.append(el('dt', {}, key), el('dd', {}, String(value)))
  }
  return dl
}

export function warningBox(message: string): HTMLElement {
  return el('div', { className: 'warning' }, message)
}
