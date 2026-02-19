namespace TorchNew {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0, size)
        return arr
    }

    export class Embedding {
        vocabSize: number
        embedDim: number
        weight: Tensor
        dW: Tensor
        lastIndices:Tensor;

        constructor(vocabSize: number, embedDim: number) {
            this.vocabSize = vocabSize
            this.embedDim = embedDim

            // Weight shape: [vocabSize, embedDim]
            let wsize = vocabSize * embedDim
            let wdata = alloc(wsize)

            // Xavier-like init
            let scale = 1 / Math.sqrt(vocabSize)
            let i = 0
            while (i < wsize) {
                wdata[i] = (Math.random() - 0.5) * 2 * scale
                i++
            }

            this.weight = new Tensor(wdata, [vocabSize, embedDim])

            // Gradient buffer
            let dwdata = alloc(wsize)
            this.dW = new Tensor(dwdata, [vocabSize, embedDim])
        }

        // Forward: indices -> embedding vectors
        forward(indices: Tensor): Tensor {
            // indices is an integer tensor (flat or ND)
            // output shape = [..., embedDim]

            let outShape = alloc(indices.shape.length + 1)
            let i = 0
            while (i < indices.shape.length) {
                outShape[i] = indices.shape[i]
                i++
            }
            outShape[indices.shape.length] = this.embedDim

            let total = 1
            i = 0
            while (i < outShape.length) {
                total *= outShape[i]
                i++
            }

            let outData = alloc(total)
            let out = new Tensor(outData, outShape)

            // Save indices for backward
            this.lastIndices = indices

            // Fill output
            let idxCount = indices.data.length
            let j = 0
            while (j < idxCount) {
                let token = indices.data[j]

                // Copy weight[token] into out[j]
                let baseOut = j * this.embedDim
                let baseW = token * this.embedDim

                let k = 0
                while (k < this.embedDim) {
                    outData[baseOut + k] = this.weight.data[baseW + k]
                    k++
                }

                j++
            }

            return out
        }

        // Backward: accumulate gradients into dW
        backward(gradOut: Tensor): Tensor {
            // gradOut shape: [..., embedDim]
            // dW[token] += gradOut[i]

            // Zero dW
            let sizeDW = this.dW.data.length
            let i = 0
            while (i < sizeDW) {
                this.dW.data[i] = 0
                i++
            }

            let indices = this.lastIndices
            let count = indices.data.length

            let j = 0
            while (j < count) {
                let token = indices.data[j]

                let baseGrad = j * this.embedDim
                let baseDW = token * this.embedDim

                let k = 0
                while (k < this.embedDim) {
                    this.dW.data[baseDW + k] += gradOut.data[baseGrad + k]
                    k++
                }

                j++
            }

            // Embedding has no gradient wrt input indices
            return null
        }
    }
}