namespace TorchNew {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0, size)
        return arr
    }

    export class Flatten {
        lastShape: number[]

        constructor() {
            // No parameters
        }

        // Forward: reshape input to 2D [batch, -1]
        forward(x: Tensor): Tensor {
            this.lastShape = x.shape.slice(0)

            let batch = x.shape[0]

            // Compute flattened size
            let flatSize = 1
            let i = 1
            while (i < x.shape.length) {
                flatSize *= x.shape[i]
                i++
            }

            // Output shape: [batch, flatSize]
            let outShape = [batch, flatSize]

            // Data is already flat, so we just reuse it
            let outData = alloc(x.data.length)
            let j = 0
            while (j < x.data.length) {
                outData[j] = x.data[j]
                j++
            }

            return new Tensor(outData, outShape)
        }

        // Backward: reshape gradient back to original shape
        backward(gradOut: Tensor): Tensor {
            // gradOut shape: [batch, flatSize]
            // We reshape back to lastShape

            let total = 1
            let i = 0
            while (i < this.lastShape.length) {
                total *= this.lastShape[i]
                i++
            }

            let gradData = alloc(total)
            let j = 0
            while (j < total) {
                gradData[j] = gradOut.data[j]
                j++
            }

            return new Tensor(gradData, this.lastShape.slice(0))
        }
    }
}