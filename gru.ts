namespace TorchNew {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0, size)
        return arr
    }

    function sigmoid(x: number): number {
        return 1 / (1 + Math.exp(-x))
    }

    function fastTanh(x: number): number {
        let e2x = Math.exp(2 * x)
        return (e2x - 1) / (e2x + 1)
    }

    export class GRU {
        inputDim: number
        hiddenDim: number

        // Parameters
        Wz: Tensor; Uz: Tensor; bz: Tensor
        Wr: Tensor; Ur: Tensor; br: Tensor
        Wh: Tensor; Uh: Tensor; bh: Tensor

        // Gradients
        dWz: Tensor; dUz: Tensor; dBz: Tensor
        dWr: Tensor; dUr: Tensor; dBr: Tensor
        dWh: Tensor; dUh: Tensor; dBh: Tensor

        // Saved for backward
        lastInput: Tensor
        lastZ: number[]
        lastR: number[]
        lastHtilde: number[]
        lastH: number[]   // includes h0

        constructor(inputDim: number, hiddenDim: number) {
            this.inputDim = inputDim
            this.hiddenDim = hiddenDim

            this.Wz = this.initWeight(hiddenDim, inputDim)
            this.Uz = this.initWeight(hiddenDim, hiddenDim)
            this.bz = new Tensor(alloc(hiddenDim), [hiddenDim])

            this.Wr = this.initWeight(hiddenDim, inputDim)
            this.Ur = this.initWeight(hiddenDim, hiddenDim)
            this.br = new Tensor(alloc(hiddenDim), [hiddenDim])

            this.Wh = this.initWeight(hiddenDim, inputDim)
            this.Uh = this.initWeight(hiddenDim, hiddenDim)
            this.bh = new Tensor(alloc(hiddenDim), [hiddenDim])

            this.dWz = this.initZero(hiddenDim, inputDim)
            this.dUz = this.initZero(hiddenDim, hiddenDim)
            this.dBz = new Tensor(alloc(hiddenDim), [hiddenDim])

            this.dWr = this.initZero(hiddenDim, inputDim)
            this.dUr = this.initZero(hiddenDim, hiddenDim)
            this.dBr = new Tensor(alloc(hiddenDim), [hiddenDim])

            this.dWh = this.initZero(hiddenDim, inputDim)
            this.dUh = this.initZero(hiddenDim, hiddenDim)
            this.dBh = new Tensor(alloc(hiddenDim), [hiddenDim])
        }

        initWeight(rows: number, cols: number): Tensor {
            let size = rows * cols
            let data = alloc(size)
            let scale = 1 / Math.sqrt(cols)
            let i = 0
            while (i < size) {
                data[i] = (Math.random() - 0.5) * 2 * scale
                i++
            }
            return new Tensor(data, [rows, cols])
        }

        initZero(rows: number, cols: number): Tensor {
            return new Tensor(alloc(rows * cols), [rows, cols])
        }

        // ---------------------------------------------------------
        // Forward
        // ---------------------------------------------------------
        forward(x: Tensor): Tensor {
            // x: [batch, seq, inputDim]
            this.lastInput = x

            let batch = x.shape[0]
            let seq = x.shape[1]
            let inDim = this.inputDim
            let hDim = this.hiddenDim

            let outData = alloc(batch * seq * hDim)

            this.lastZ = alloc(batch * seq * hDim)
            this.lastR = alloc(batch * seq * hDim)
            this.lastHtilde = alloc(batch * seq * hDim)
            this.lastH = alloc(batch * (seq + 1) * hDim)

            // h0 = 0
            let i = 0
            while (i < batch * hDim) {
                this.lastH[i] = 0
                i++
            }

            let t = 0
            while (t < seq) {
                let b = 0
                while (b < batch) {

                    let baseX = (b * seq + t) * inDim
                    let baseHprev = (b * (seq + 1) + t) * hDim
                    let baseHcur = (b * (seq + 1) + (t + 1)) * hDim

                    let baseZ = (b * seq + t) * hDim
                    let baseR = (b * seq + t) * hDim
                    let baseHtilde = (b * seq + t) * hDim

                    // ---- Compute z_t ----
                    let j = 0
                    while (j < hDim) {
                        let sum = this.bz.data[j]

                        let k = 0
                        while (k < inDim) {
                            sum += this.Wz.data[j * inDim + k] * x.data[baseX + k]
                            k++
                        }

                        k = 0
                        while (k < hDim) {
                            sum += this.Uz.data[j * hDim + k] * this.lastH[baseHprev + k]
                            k++
                        }

                        let z = sigmoid(sum)
                        this.lastZ[baseZ + j] = z
                        j++
                    }

                    // ---- Compute r_t ----
                    j = 0
                    while (j < hDim) {
                        let sum = this.br.data[j]

                        let k = 0
                        while (k < inDim) {
                            sum += this.Wr.data[j * inDim + k] * x.data[baseX + k]
                            k++
                        }

                        k = 0
                        while (k < hDim) {
                            sum += this.Ur.data[j * hDim + k] * this.lastH[baseHprev + k]
                            k++
                        }

                        let r = sigmoid(sum)
                        this.lastR[baseR + j] = r
                        j++
                    }

                    // ---- Compute h~_t ----
                    j = 0
                    while (j < hDim) {
                        let sum = this.bh.data[j]

                        let k = 0
                        while (k < inDim) {
                            sum += this.Wh.data[j * inDim + k] * x.data[baseX + k]
                            k++
                        }

                        k = 0
                        while (k < hDim) {
                            let hprev = this.lastH[baseHprev + k]
                            let r = this.lastR[baseR + k]
                            sum += this.Uh.data[j * hDim + k] * (r * hprev)
                            k++
                        }

                        let htilde = fastTanh(sum)
                        this.lastHtilde[baseHtilde + j] = htilde
                        j++
                    }

                    // ---- Final h_t ----
                    j = 0
                    while (j < hDim) {
                        let z = this.lastZ[baseZ + j]
                        let hprev = this.lastH[baseHprev + j]
                        let htilde = this.lastHtilde[baseHtilde + j]

                        let h = (1 - z) * hprev + z * htilde
                        this.lastH[baseHcur + j] = h

                        outData[(b * seq + t) * hDim + j] = h
                        j++
                    }

                    b++
                }
                t++
            }

            return new Tensor(outData, [batch, seq, hDim])
        }

        // ---------------------------------------------------------
        // Backward (BPTT)
        // ---------------------------------------------------------
        backward(gradOut: Tensor): Tensor {
            let x = this.lastInput
            let batch = x.shape[0]
            let seq = x.shape[1]
            let inDim = this.inputDim
            let hDim = this.hiddenDim

            // Zero gradients
            let i = 0
            while (i < this.dWz.data.length) { this.dWz.data[i] = 0; i++ }
            i = 0
            while (i < this.dUz.data.length) { this.dUz.data[i] = 0; i++ }
            i = 0
            while (i < this.dBz.data.length) { this.dBz.data[i] = 0; i++ }

            i = 0
            while (i < this.dWr.data.length) { this.dWr.data[i] = 0; i++ }
            i = 0
            while (i < this.dUr.data.length) { this.dUr.data[i] = 0; i++ }
            i = 0
            while (i < this.dBr.data.length) { this.dBr.data[i] = 0; i++ }

            i = 0
            while (i < this.dWh.data.length) { this.dWh.data[i] = 0; i++ }
            i = 0
            while (i < this.dUh.data.length) { this.dUh.data[i] = 0; i++ }
            i = 0
            while (i < this.dBh.data.length) { this.dBh.data[i] = 0; i++ }

            let dXdata = alloc(x.data.length)
            let dHnext = alloc(batch * hDim)

            let t = seq - 1
            while (t >= 0) {
                let b = 0
                while (b < batch) {

                    let baseX = (b * seq + t) * inDim
                    let baseHprev = (b * (seq + 1) + t) * hDim
                    let baseHcur = (b * (seq + 1) + (t + 1)) * hDim
                    let baseZ = (b * seq + t) * hDim
                    let baseR = (b * seq + t) * hDim
                    let baseHtilde = (b * seq + t) * hDim

                    // dL/dh_t
                    let dH = alloc(hDim)
                    let j = 0
                    while (j < hDim) {
                        dH[j] = gradOut.data[(b * seq + t) * hDim + j] + dHnext[b * hDim + j]
                        j++
                    }

                    // Compute gate derivatives
                    let dZ = alloc(hDim)
                    let dHtilde = alloc(hDim)
                    let dHprev = alloc(hDim)

                    j = 0
                    while (j < hDim) {
                        let z = this.lastZ[baseZ + j]
                        let hprev = this.lastH[baseHprev + j]
                        let htilde = this.lastHtilde[baseHtilde + j]

                        dZ[j] = dH[j] * (htilde - hprev)
                        dHtilde[j] = dH[j] * z
                        dHprev[j] = dH[j] * (1 - z)
                        j++
                    }

                    // dA_h = dHtilde * (1 - htilde^2)
                    let dA_h = alloc(hDim)
                    j = 0
                    while (j < hDim) {
                        let htilde = this.lastHtilde[baseHtilde + j]
                        dA_h[j] = dHtilde[j] * (1 - htilde * htilde)
                        j++
                    }

                    // dA_z = dZ * z * (1 - z)
                    let dA_z = alloc(hDim)
                    j = 0
                    while (j < hDim) {
                        let z = this.lastZ[baseZ + j]
                        dA_z[j] = dZ[j] * z * (1 - z)
                        j++
                    }

                    // dA_r = dHtilde * Uh * hprev * r*(1-r)
                    let dA_r = alloc(hDim)
                    j = 0
                    while (j < hDim) {
                        let sum = 0
                        let k = 0
                        while (k < hDim) {
                            let w = this.Uh.data[k * hDim + j]
                            let d = dA_h[k]
                            sum += d * w
                            k++
                        }
                        let r = this.lastR[baseR + j]
                        let hprev = this.lastH[baseHprev + j]
                        dA_r[j] = sum * hprev * r * (1 - r)
                        j++
                    }

                    // ---- Accumulate gradients ----

                    // dWh, dUh, dBh
                    j = 0
                    while (j < hDim) {
                        this.dBh.data[j] += dA_h[j]

                        let k = 0
                        while (k < inDim) {
                            this.dWh.data[j * inDim + k] += dA_h[j] * x.data[baseX + k]
                            k++
                        }

                        k = 0
                        while (k < hDim) {
                            let r = this.lastR[baseR + k]
                            let hprev = this.lastH[baseHprev + k]
                            this.dUh.data[j * hDim + k] += dA_h[j] * (r * hprev)
                            k++
                        }

                        j++
                    }

                    // dWz, dUz, dBz
                    j = 0
                    while (j < hDim) {
                        this.dBz.data[j] += dA_z[j]

                        let k = 0
                        while (k < inDim) {
                            this.dWz.data[j * inDim + k] += dA_z[j] * x.data[baseX + k]
                            k++
                        }

                        k = 0
                        while (k < hDim) {
                            this.dUz.data[j * hDim + k] += dA_z[j] * this.lastH[baseHprev + k]
                            k++
                        }

                        j++
                    }

                    // dWr, dUr, dBr
                    j = 0
                    while (j < hDim) {
                        this.dBr.data[j] += dA_r[j]

                        let k = 0
                        while (k < inDim) {
                            this.dWr.data[j * inDim + k] += dA_r[j] * x.data[baseX + k]
                            k++
                        }

                        k = 0
                        while (k < hDim) {
                            this.dUr.data[j * hDim + k] += dA_r[j] * this.lastH[baseHprev + k]
                            k++
                        }

                        j++
                    }

                    // ---- Compute dX ----
                    j = 0
                    while (j < inDim) {
                        let sum = 0

                        let k = 0
                        while (k < hDim) {
                            sum += dA_z[k] * this.Wz.data[k * inDim + j]
                            sum += dA_r[k] * this.Wr.data[k * inDim + j]
                            sum += dA_h[k] * this.Wh.data[k * inDim + j]
                            k++
                        }

                        dXdata[baseX + j] = sum
                        j++
                    }

                    // ---- Compute dHnext ----
                    j = 0
                    while (j < hDim) {
                        let sum = dHprev[j]

                        let k = 0
                        while (k < hDim) {
                            sum += dA_z[k] * this.Uz.data[k * hDim + j]
                            sum += dA_r[k] * this.Ur.data[k * hDim + j]
                            sum += dA_h[k] * this.Uh.data[k * hDim + j] * this.lastR[baseR + j]
                            k++
                        }

                        dHnext[b * hDim + j] = sum
                        j++
                    }

                    b++
                }

                t--
            }

            return new Tensor(dXdata, x.shape.slice(0))
        }
    }
}