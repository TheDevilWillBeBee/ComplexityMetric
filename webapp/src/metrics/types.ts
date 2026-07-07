import type { EmbeddingData } from '../extractors/types'

/**
 * A complexity metric scores each batch element of a (B, T, D) embedding.
 * `compute` assumes the input has already been converted to the metric's
 * input type (see convert.ts / ComplexityMetric.__call__ in base.py).
 */
export interface Metric {
  inputType: 'discrete' | 'continuous'
  compute(e: EmbeddingData): Promise<Float64Array>
}
