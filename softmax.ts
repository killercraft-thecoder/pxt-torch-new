namespace TorchNew.Activations {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0, size)
        return arr
    }

    export class Softmax {

        // Forward: softmax along last dimension
        forward(x: Tensor): Tensor {
            let size = x.data.length
            let lastDim = x.shape[x.shape.length - 1]

            let outData = alloc(size)

            let i = 0
            while (i < size) {
                // Compute index of start of this row
                let rowStart = i - (i % lastDim)

                // 1. Find max for numerical stability
                let maxVal = x.data[rowStart]
                let j = 1
                while (j < lastDim) {
                    let v = x.data[rowStart + j]
                    if (v > maxVal) maxVal = v
                    j++
                }

                // 2. Compute exp(x - max)
                let sumExp = 0
                j = 0
                while (j < lastDim) {
                    let e = Math.exp(x.data[rowStart + j] - maxVal)
                    outData[rowStart + j] = e
                    sumExp += e
                    j++
                }

                // 3. Normalize
                j = 0
                while (j < lastDim) {
                    outData[rowStart + j] /= sumExp
                    j++
                }

                // Move to next row
                i = rowStart + lastDim
            }

            return new Tensor(outData, x.shape.slice(0))
        }

        // Backward: dSoftmax = softmax * (gradOut - sum(gradOut * softmax))
        backward(x: Tensor, gradOut: Tensor): Tensor {
            let size = x.data.length
            let lastDim = x.shape[x.shape.length - 1]

            let grad = alloc(size)

            // First compute softmax(x)
            let soft = this.forward(x)

            let i = 0
            while (i < size) {
                let rowStart = i - (i % lastDim)

                // Compute dot(gradOut, softmax)
                let dot = 0
                let j = 0
                while (j < lastDim) {
                    dot += gradOut.data[rowStart + j] * soft.data[rowStart + j]
                    j++
                }

                // Compute gradient
                j = 0
                while (j < lastDim) {
                    let s = soft.data[rowStart + j]
                    grad[rowStart + j] = s * (gradOut.data[rowStart + j] - dot)
                    j++
                }

                i = rowStart + lastDim
            }

            return new Tensor(grad, x.shape.slice(0))
        }
    }
}