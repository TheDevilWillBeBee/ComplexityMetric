/** Birth/survival rule-string parsing (DynamicalSystems/utils.py:88-112). */

export interface BsSets {
  birth: Set<number>
  survive: Set<number>
}

/**
 * Parse "B3/S23"-style strings. Digits are taken per-character unless a
 * separator (, ; : _ -) is present, in which case multi-digit numbers are
 * parsed (needed for neighbor sums >= 10 with large kernels).
 */
export function parseBs(desc: string): BsSets {
  const s = desc.trim().toUpperCase().replace(/ /g, '')
  const mB = /B([^S]*)/.exec(s)
  const mS = /S(.*)/.exec(s)
  const nums = (part: string): Set<number> => {
    if (/[,;:_-]/.test(part)) {
      return new Set((part.match(/\d+/g) ?? []).map(Number))
    }
    return new Set([...part].filter((ch) => /\d/.test(ch)).map(Number))
  }
  return {
    birth: mB ? nums(mB[1]) : new Set(),
    survive: mS ? nums(mS[1]) : new Set(),
  }
}

/** Lookup tables of length L: table[n] = 1 iff n is in the set. */
export function bsTables(sets: BsSets, L: number): { birth: Uint8Array; survive: Uint8Array } {
  const birth = new Uint8Array(L)
  const survive = new Uint8Array(L)
  for (const n of sets.birth) if (n >= 0 && n < L) birth[n] = 1
  for (const n of sets.survive) if (n >= 0 && n < L) survive[n] = 1
  return { birth, survive }
}

/** Random tables with independent Bernoulli densities (systems.py sample_params). */
export function randomBsTables(
  L: number,
  pBirth: number,
  pSurvive: number,
  rand: () => number,
): { birth: Uint8Array; survive: Uint8Array } {
  const birth = new Uint8Array(L)
  const survive = new Uint8Array(L)
  for (let i = 0; i < L; i++) birth[i] = rand() < pBirth ? 1 : 0
  for (let i = 0; i < L; i++) survive[i] = rand() < pSurvive ? 1 : 0
  return { birth, survive }
}
