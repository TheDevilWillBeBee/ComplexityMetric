declare module 'tsne-js' {
  export interface TSNEOptions {
    dim?: number
    perplexity?: number
    earlyExaggeration?: number
    learningRate?: number
    nIter?: number
    metric?: string
  }
  export default class TSNE {
    constructor(options?: TSNEOptions)
    init(input: { data: number[][]; type: 'dense' | 'sparse' }): void
    run(): [number, number]
    getOutput(): number[][]
    getOutputScaled(): number[][]
  }
}
