namespace TorchNew {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0, size)
        return arr
    }

    export class Adam {
        lr: number
        beta1: number
        beta2: number
        eps: number
        t: number

        // Moment buffers: arrays of Tensors
        m: Tensor[]
        v: Tensor[]

        constructor(lr: number, beta1: number, beta2: number, eps: number) {
            this.lr = lr
            this.beta1 = beta1
            this.beta2 = beta2
            this.eps = eps
            this.t = 0

            this.m = []
            this.v = []
        }

        // Initialize moment buffers for a parameter list
        init(params: Tensor[]): void {
            let i = 0
            while (i < params.length) {
                let p = params[i]

                let size = p.data.length
                let mdata = alloc(size)
                let vdata = alloc(size)

                this.m.push(new Tensor(mdata, p.shape.slice(0)))
                this.v.push(new Tensor(vdata, p.shape.slice(0)))

                i++
            }
        }

        // Apply Adam update to a list of (param, grad) pairs
        step(params: Tensor[], grads: Tensor[]): void {
            this.t++

            let lr = this.lr
            let b1 = this.beta1
            let b2 = this.beta2
            let eps = this.eps

            let t = this.t

            let i = 0
            while (i < params.length) {
                let p = params[i]
                let g = grads[i]
                let m = this.m[i]
                let v = this.v[i]

                let size = p.data.length
                let j = 0

                // Update each element
                while (j < size) {
                    let grad = g.data[j]

                    // m_t = b1*m + (1-b1)*g
                    m.data[j] = b1 * m.data[j] + (1 - b1) * grad

                    // v_t = b2*v + (1-b2)*g^2
                    v.data[j] = b2 * v.data[j] + (1 - b2) * grad * grad

                    // Bias correction
                    let mHat = m.data[j] / (1 - Math.pow(b1, t))
                    let vHat = v.data[j] / (1 - Math.pow(b2, t))

                    // Parameter update
                    p.data[j] -= lr * mHat / (Math.sqrt(vHat) + eps)

                    j++
                }

                i++
            }
        }
    }
}