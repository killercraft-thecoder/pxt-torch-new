namespace TorchNew {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0, size)
        return arr
    }

    export class Dropout {
        p: number
        training: boolean
        lastMask: number[]

        constructor(p: number) {
            this.p = p
            this.training = true
        }

        // Enable/disable training mode
        train(): void {
            this.training = true
        }

        eval(): void {
            this.training = false
        }

        // ---------------------------------------------------------
        // Forward
        // ---------------------------------------------------------
        forward(x: Tensor): Tensor {
            let size = x.data.length
            let outData = alloc(size)

            // If not training, dropout does nothing
            if (!this.training) {
                let i = 0
                while (i < size) {
                    outData[i] = x.data[i]
                    i++
                }
                return new Tensor(outData, x.shape.slice(0))
            }

            // Training mode: generate mask
            this.lastMask = alloc(size)

            let scale = 1 / (1 - this.p)
            let i = 0
            while (i < size) {
                // 1 = keep, 0 = drop
                let keep = Math.random() > this.p ? 1 : 0
                this.lastMask[i] = keep

                outData[i] = x.data[i] * keep * scale
                i++
            }

            return new Tensor(outData, x.shape.slice(0))
        }

        // ---------------------------------------------------------
        // Backward
        // ---------------------------------------------------------
        backward(gradOut: Tensor): Tensor {
            let size = gradOut.data.length
            let gradData = alloc(size)

            // If not training, gradient passes unchanged
            if (!this.training) {
                let i = 0
                while (i < size) {
                    gradData[i] = gradOut.data[i]
                    i++
                }
                return new Tensor(gradData, gradOut.shape.slice(0))
            }

            // Training mode: apply mask
            let scale = 1 / (1 - this.p)
            let i = 0
            while (i < size) {
                gradData[i] = gradOut.data[i] * this.lastMask[i] * scale
                i++
            }

            return new Tensor(gradData, gradOut.shape.slice(0))
        }
    }
}