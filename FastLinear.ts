namespace TorchNew {

    export class FastLinear {
        inFeatures: number
        outFeatures: number
        weight: TorchNew.FastTensor
        bias: TorchNew.FastTensor
        lastInput:FastTensor;

        constructor(inFeatures: number, outFeatures: number) {
            this.inFeatures = inFeatures
            this.outFeatures = outFeatures

            // Weight: [outFeatures][inFeatures]
            let w: number[][] = []
            for (let r = 0; r < outFeatures; r++) {
                let row: number[] = []
                for (let c = 0; c < inFeatures; c++) {
                    // Xavier-like init
                    row.push((Math.random() - 0.5) * 2 / Math.sqrt(inFeatures))
                }
                w.push(row)
            }
            this.weight = new TorchNew.FastTensor(w)

            // Bias: [outFeatures][1] (or just a row vector)
            let b: number[][] = []
            let brow: number[] = []
            for (let i = 0; i < outFeatures; i++) brow.push(0)
            b.push(brow)
            this.bias = new TorchNew.FastTensor(b)
        }

        // Forward: X (batch × inFeatures) * W^T (inFeatures × outFeatures) + b
        forward(x: TorchNew.FastTensor): TorchNew.FastTensor {
            // x.data: [batch][inFeatures]
            // weight.data: [outFeatures][inFeatures]
            // but matmul expects: A(rowsA × colsA) * B(rowsB × colsB)
            // so we need weight^T: [inFeatures][outFeatures]

            this.lastInput = x

            let Wt = this.transpose2D(this.weight.data)

            // matmul: (batch × inFeatures) * (inFeatures × outFeatures)
            let out = x.matmul(new TorchNew.FastTensor(Wt)) as TorchNew.FastTensor

            // Add bias row-wise
            let batch = out.data.length
            let outF = this.outFeatures

            for (let r = 0; r < batch; r++) {
                for (let c = 0; c < outF; c++) {
                    out.data[r][c] += this.bias.data[0][c]
                }
            }

            return out
        }

        backward(dY: TorchNew.FastTensor, x: TorchNew.FastTensor) {
            // dx = dY @ W
            let dx = dY.matmul(this.weight) as TorchNew.FastTensor

            // dW = dY^T @ x
            let dYt = new TorchNew.FastTensor(this.transpose2D(dY.data))
            let dW = dYt.matmul(x) as TorchNew.FastTensor

            // db = sum over batch rows of dY
            let dbRow: number[] = []
            for (let i = 0; i < this.outFeatures; i++) dbRow.push(0)

            let batch = dY.data.length
            for (let r = 0; r < batch; r++) {
                for (let c = 0; c < this.outFeatures; c++) {
                    dbRow[c] += dY.data[r][c]
                }
            }

            let db = new TorchNew.FastTensor([dbRow])

            return {
                dx: dx,
                dW: dW,
                db: db
            }
        }



        // Simple 2D transpose helper
        private transpose2D(m: number[][]): number[][] {
            let rows = m.length
            let cols = m[0].length
            let out: number[][] = []

            for (let c = 0; c < cols; c++) {
                let row: number[] = []
                for (let r = 0; r < rows; r++) {
                    row.push(m[r][c])
                }
                out.push(row)
            }
            return out
        }
    }
}