namespace TorchNew {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0, size)
        return arr
    }

    export class Linear {
        inFeatures: number
        outFeatures: number
        weight: Tensor
        bias: Tensor

        constructor(inFeatures: number, outFeatures: number) {
            this.inFeatures = inFeatures
            this.outFeatures = outFeatures

            // Weight shape: [outFeatures, inFeatures]
            let wsize = outFeatures * inFeatures
            let wdata = alloc(wsize)

            // Xavier-like init (simple version)
            let i = 0
            while (i < wsize) {
                wdata[i] = (Math.random() - 0.5) * 2 / Math.sqrt(inFeatures)
                i++
            }

            this.weight = new Tensor(wdata, [outFeatures, inFeatures])

            // Bias shape: [outFeatures]
            let bdata = alloc(outFeatures)
            i = 0
            while (i < outFeatures) {
                bdata[i] = 0
                i++
            }

            this.bias = new Tensor(bdata, [outFeatures])
        }

        // Forward pass: x @ W^T + b
        forward(x: Tensor): Tensor {
            // x shape: [..., inFeatures]
            // weight shape: [outFeatures, inFeatures]
            // Need W^T shape: [inFeatures, outFeatures]

            let Wt = this.weight.transpose(0, 1)

            // Matmul: [..., inFeatures] × [inFeatures, outFeatures]
            let out = x.matmul(Wt)

            // Add bias (broadcast over batch dims)
            let size = out.data.length
            let lastDim = this.outFeatures
            let i = 0

            while (i < size) {
                let idx = i % lastDim
                out.data[i] += this.bias.data[idx]
                i++
            }

            return out
        }

        backward(dY: Tensor, x: Tensor): { dx: Tensor, dW: Tensor, db: Tensor } {
            // dY shape: [..., outFeatures]
            // x shape:  [..., inFeatures]
            // W shape:  [outFeatures, inFeatures]

            // 1. dx = dY @ W
            let dx = dY.matmul(this.weight)

            // 2. dW = dY^T @ x
            let dYt = dY.transpose(dY.shape.length - 2, dY.shape.length - 1)
            let dW = dYt.matmul(x)

            // 3. db = sum(dY over all batch dims)
            let dbData = alloc(this.outFeatures)
            let size = dY.data.length
            let lastDim = this.outFeatures
            let i = 0

            while (i < size) {
                let idx = i % lastDim
                dbData[idx] += dY.data[i]
                i++
            }

            let db = new Tensor(dbData, [this.outFeatures])

            return {
                dx: dx,
                dW: dW,
                db: db
            }
        }
    }
}