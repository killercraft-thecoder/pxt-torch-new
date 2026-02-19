// Add your code here
namespace TorchNew {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0, size)
        return arr
    }

    export class PositionalEncoding {
        maxLen: number
        dim: number
        encoding: Tensor

        constructor(maxLen: number, dim: number) {
            this.maxLen = maxLen
            this.dim = dim

            // Precompute positional encodings
            let data = alloc(maxLen * dim)

            let pos = 0
            while (pos < maxLen) {
                let i = 0
                while (i < dim) {
                    let angle = pos / Math.pow(10000, (2 * Math.idiv(i, 2)) / dim)

                    if (i % 2 == 0) {
                        data[pos * dim + i] = Math.sin(angle)
                    } else {
                        data[pos * dim + i] = Math.cos(angle)
                    }

                    i++
                }
                pos++
            }

            this.encoding = new Tensor(data, [maxLen, dim])
        }

        // ---------------------------------------------------------
        // Forward: add positional encoding to input
        // x shape: [batch, seq, dim]
        // ---------------------------------------------------------
        forward(x: Tensor): Tensor {
            let batch = x.shape[0]
            let seq = x.shape[1]
            let dim = x.shape[2]

            let outData = alloc(x.data.length)

            let i = 0
            while (i < batch) {
                let j = 0
                while (j < seq) {
                    let k = 0
                    while (k < dim) {
                        let idx = i * seq * dim + j * dim + k
                        let pe = this.encoding.data[j * dim + k]
                        outData[idx] = x.data[idx] + pe
                        k++
                    }
                    j++
                }
                i++
            }

            return new Tensor(outData, x.shape.slice(0))
        }

        // ---------------------------------------------------------
        // Backward: gradient passes through unchanged
        // ---------------------------------------------------------
        backward(gradOut: Tensor): Tensor {
            let gradData = alloc(gradOut.data.length)
            let i = 0
            while (i < gradOut.data.length) {
                gradData[i] = gradOut.data[i]
                i++
            }
            return new Tensor(gradData, gradOut.shape.slice(0))
        }
    }
}