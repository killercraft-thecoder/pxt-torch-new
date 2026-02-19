namespace TorchNew.Activations {

    export interface FastActivation {
        forward(x: FastTensor): FastTensor
        backward(x: FastTensor, gradOut: FastTensor): FastTensor
    }

    // ---------------------------------------------------------
    // ReLU
    // ---------------------------------------------------------
    export class FastReLU implements FastActivation {

        forward(x: FastTensor): FastTensor {
            // Store input for backward
            this.lastInput = x
            return x.applyFunction(v => v > 0 ? v : 0)
        }

        backward(x: FastTensor, gradOut: FastTensor): FastTensor {
            return new FastTensor(
                x.data.map((row, r) =>
                    row.map((v, c) => v > 0 ? gradOut.data[r][c] : 0)
                )
            )
        }

        lastInput: FastTensor
    }

    // ---------------------------------------------------------
    // Sigmoid
    // ---------------------------------------------------------
    export class FastSigmoid implements FastActivation {

        forward(x: FastTensor): FastTensor {
            this.lastInput = x
            return x.applyFunction(v => 1 / (1 + Math.exp(-v)))
        }

        backward(x: FastTensor, gradOut: FastTensor): FastTensor {
            let out: number[][] = []
            for (let r = 0; r < x.data.length; r++) {
                let row: number[] = []
                for (let c = 0; c < x.data[0].length; c++) {
                    let s = 1 / (1 + Math.exp(-x.data[r][c]))
                    row.push(gradOut.data[r][c] * s * (1 - s))
                }
                out.push(row)
            }
            return new FastTensor(out)
        }

        lastInput: FastTensor
    }

    // ---------------------------------------------------------
    // Tanh
    // ---------------------------------------------------------
    export class FastTanh implements FastActivation {

        forward(x: FastTensor): FastTensor {
            this.lastInput = x
            return x.applyFunction(v => {
                let e2 = Math.exp(2 * v)
                return (e2 - 1) / (e2 + 1)
            })
        }

        backward(x: FastTensor, gradOut: FastTensor): FastTensor {
            let out: number[][] = []
            for (let r = 0; r < x.data.length; r++) {
                let row: number[] = []
                for (let c = 0; c < x.data[0].length; c++) {
                    let v = x.data[r][c]
                    let e2 = Math.exp(2 * v)
                    let t = (e2 - 1) / (e2 + 1)
                    row.push(gradOut.data[r][c] * (1 - t * t))
                }
                out.push(row)
            }
            return new FastTensor(out)
        }

        lastInput: FastTensor
    }

    // ---------------------------------------------------------
    // LeakyReLU
    // ---------------------------------------------------------
    export class FastLeakyReLU implements FastActivation {
        alpha: number
        lastInput: FastTensor

        constructor(alpha: number) {
            this.alpha = alpha
        }

        forward(x: FastTensor): FastTensor {
            this.lastInput = x
            return x.applyFunction(v => v > 0 ? v : this.alpha * v)
        }

        backward(x: FastTensor, gradOut: FastTensor): FastTensor {
            let out: number[][] = []
            for (let r = 0; r < x.data.length; r++) {
                let row: number[] = []
                for (let c = 0; c < x.data[0].length; c++) {
                    let v = x.data[r][c]
                    row.push(v > 0 ? gradOut.data[r][c] : this.alpha * gradOut.data[r][c])
                }
                out.push(row)
            }
            return new FastTensor(out)
        }
    }
}