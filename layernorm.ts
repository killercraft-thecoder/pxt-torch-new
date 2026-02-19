namespace TorchNew {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0, size)
        return arr
    }

    export class LayerNorm {
        normalizedShape: number[]
        eps: number

        gamma: Tensor
        beta: Tensor

        dGamma: Tensor
        dBeta: Tensor

        lastInput: Tensor
        lastMean: number[]
        lastVar: number[]
        lastNorm: number[]

        constructor(normalizedShape: number[], eps: number = 1e-5) {
            this.normalizedShape = normalizedShape.slice(0)
            this.eps = eps

            // Parameter size = product of normalizedShape
            let size = 1
            let i = 0
            while (i < normalizedShape.length) {
                size *= normalizedShape[i]
                i++
            }

            // gamma initialized to 1
            let gdata = alloc(size)
            i = 0
            while (i < size) {
                gdata[i] = 1
                i++
            }
            this.gamma = new Tensor(gdata, normalizedShape.slice(0))

            // beta initialized to 0
            let bdata = alloc(size)
            this.beta = new Tensor(bdata, normalizedShape.slice(0))

            // gradients
            this.dGamma = new Tensor(alloc(size), normalizedShape.slice(0))
            this.dBeta = new Tensor(alloc(size), normalizedShape.slice(0))
        }

        // ---------------------------------------------------------
        // Forward
        // ---------------------------------------------------------
        forward(x: Tensor): Tensor {
            this.lastInput = x

            let dims = this.normalizedShape.length
            let total = x.data.length

            // Compute size of the normalized block
            let blockSize = 1
            let i = 0
            while (i < dims) {
                blockSize *= this.normalizedShape[i]
                i++
            }

            // Number of blocks = total / blockSize
            let blocks = Math.idiv(total, blockSize)

            let outData = alloc(total)
            let out = new Tensor(outData, x.shape.slice(0))

            this.lastMean = alloc(blocks)
            this.lastVar = alloc(blocks)
            this.lastNorm = alloc(total)

            let b = 0
            while (b < blocks) {
                let start = b * blockSize

                // Compute mean
                let mean = 0
                let j = 0
                while (j < blockSize) {
                    mean += x.data[start + j]
                    j++
                }
                mean /= blockSize
                this.lastMean[b] = mean

                // Compute variance
                let variance = 0
                j = 0
                while (j < blockSize) {
                    let d = x.data[start + j] - mean
                    variance += d * d
                    j++
                }
                variance /= blockSize
                this.lastVar[b] = variance

                let invStd = 1 / Math.sqrt(variance + this.eps)

                // Normalize + scale + shift
                j = 0
                while (j < blockSize) {
                    let norm = (x.data[start + j] - mean) * invStd
                    this.lastNorm[start + j] = norm

                    outData[start + j] =
                        norm * this.gamma.data[j] + this.beta.data[j]

                    j++
                }

                b++
            }

            return out
        }

        // ---------------------------------------------------------
        // Backward
        // ---------------------------------------------------------
        backward(gradOut: Tensor): Tensor {
            let x = this.lastInput
            let total = x.data.length

            let dims = this.normalizedShape.length

            // Compute block size
            let blockSize = 1
            let i = 0
            while (i < dims) {
                blockSize *= this.normalizedShape[i]
                i++
            }

            let blocks = Math.idiv(total, blockSize)

            // Zero gradients
            let size = this.gamma.data.length
            i = 0
            while (i < size) {
                this.dGamma.data[i] = 0
                this.dBeta.data[i] = 0
                i++
            }

            let dXdata = alloc(total)

            let b = 0
            while (b < blocks) {
                let start = b * blockSize
                let mean = this.lastMean[b]
                let variance = this.lastVar[b]
                let invStd = 1 / Math.sqrt(variance + this.eps)

                // Compute dGamma and dBeta
                let j = 0
                while (j < blockSize) {
                    let go = gradOut.data[start + j]
                    let norm = this.lastNorm[start + j]

                    this.dGamma.data[j] += go * norm
                    this.dBeta.data[j] += go

                    j++
                }

                // Compute dX
                // Formula:
                // dX = (1/N)*gamma*invStd * [N*gradOut - sum(gradOut) - norm*sum(gradOut*norm)]
                let sumGO = 0
                let sumGONorm = 0

                j = 0
                while (j < blockSize) {
                    let go = gradOut.data[start + j]
                    let norm = this.lastNorm[start + j]

                    sumGO += go
                    sumGONorm += go * norm
                    j++
                }

                j = 0
                while (j < blockSize) {
                    let go = gradOut.data[start + j]
                    let norm = this.lastNorm[start + j]

                    let term = go * blockSize - sumGO - norm * sumGONorm
                    term /= blockSize

                    dXdata[start + j] = this.gamma.data[j] * invStd * term

                    j++
                }

                b++
            }

            return new Tensor(dXdata, x.shape.slice(0))
        }
    }
}