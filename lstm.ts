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

    export class LSTM {
        inputDim: number
        hiddenDim: number

        // Parameters
        Wi: Tensor; Ui: Tensor; bi: Tensor
        Wf: Tensor; Uf: Tensor; bf: Tensor
        Wo: Tensor; Uo: Tensor; bo: Tensor
        Wc: Tensor; Uc: Tensor; bc: Tensor

        // Gradients
        dWi: Tensor; dUi: Tensor; dBi: Tensor
        dWf: Tensor; dUf: Tensor; dBf: Tensor
        dWo: Tensor; dUo: Tensor; dBo: Tensor
        dWc: Tensor; dUc: Tensor; dBc: Tensor

        // Saved for backward
        lastInput: Tensor
        lastI: number[]
        lastF: number[]
        lastO: number[]
        lastCtilde: number[]
        lastC: number[]
        lastH: number[]

        constructor(inputDim: number, hiddenDim: number) {
            this.inputDim = inputDim
            this.hiddenDim = hiddenDim

            this.Wi = this.initWeight(hiddenDim, inputDim)
            this.Ui = this.initWeight(hiddenDim, hiddenDim)
            this.bi = new Tensor(alloc(hiddenDim), [hiddenDim])

            this.Wf = this.initWeight(hiddenDim, inputDim)
            this.Uf = this.initWeight(hiddenDim, hiddenDim)
            this.bf = new Tensor(alloc(hiddenDim), [hiddenDim])

            this.Wo = this.initWeight(hiddenDim, inputDim)
            this.Uo = this.initWeight(hiddenDim, hiddenDim)
            this.bo = new Tensor(alloc(hiddenDim), [hiddenDim])

            this.Wc = this.initWeight(hiddenDim, inputDim)
            this.Uc = this.initWeight(hiddenDim, hiddenDim)
            this.bc = new Tensor(alloc(hiddenDim), [hiddenDim])

            this.dWi = this.initZero(hiddenDim, inputDim)
            this.dUi = this.initZero(hiddenDim, hiddenDim)
            this.dBi = new Tensor(alloc(hiddenDim), [hiddenDim])

            this.dWf = this.initZero(hiddenDim, inputDim)
            this.dUf = this.initZero(hiddenDim, hiddenDim)
            this.dBf = new Tensor(alloc(hiddenDim), [hiddenDim])

            this.dWo = this.initZero(hiddenDim, inputDim)
            this.dUo = this.initZero(hiddenDim, hiddenDim)
            this.dBo = new Tensor(alloc(hiddenDim), [hiddenDim])

            this.dWc = this.initZero(hiddenDim, inputDim)
            this.dUc = this.initZero(hiddenDim, hiddenDim)
            this.dBc = new Tensor(alloc(hiddenDim), [hiddenDim])
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

            this.lastI = alloc(batch * seq * hDim)
            this.lastF = alloc(batch * seq * hDim)
            this.lastO = alloc(batch * seq * hDim)
            this.lastCtilde = alloc(batch * seq * hDim)
            this.lastC = alloc(batch * (seq + 1) * hDim)
            this.lastH = alloc(batch * (seq + 1) * hDim)

            // Initialize h0 = 0, c0 = 0
            let i = 0
            while (i < batch * hDim) {
                this.lastH[i] = 0
                this.lastC[i] = 0
                i++
            }

            let t = 0
            while (t < seq) {
                let b = 0
                while (b < batch) {

                    let baseX = (b * seq + t) * inDim
                    let baseHprev = (b * (seq + 1) + t) * hDim
                    let baseCprev = (b * (seq + 1) + t) * hDim
                    let baseHcur = (b * (seq + 1) + (t + 1)) * hDim
                    let baseCcur = (b * (seq + 1) + (t + 1)) * hDim

                    let baseI = (b * seq + t) * hDim
                    let baseF = (b * seq + t) * hDim
                    let baseO = (b * seq + t) * hDim
                    let baseCtilde = (b * seq + t) * hDim

                    // ---- Compute gates ----
                    let j = 0
                    while (j < hDim) {
                        // Input gate
                        let sumI = this.bi.data[j]
                        let sumF = this.bf.data[j]
                        let sumO = this.bo.data[j]
                        let sumC = this.bc.data[j]

                        let k = 0
                        while (k < inDim) {
                            let xv = x.data[baseX + k]
                            sumI += this.Wi.data[j * inDim + k] * xv
                            sumF += this.Wf.data[j * inDim + k] * xv
                            sumO += this.Wo.data[j * inDim + k] * xv
                            sumC += this.Wc.data[j * inDim + k] * xv
                            k++
                        }

                        k = 0
                        while (k < hDim) {
                            let hprev = this.lastH[baseHprev + k]
                            sumI += this.Ui.data[j * hDim + k] * hprev
                            sumF += this.Uf.data[j * hDim + k] * hprev
                            sumO += this.Uo.data[j * hDim + k] * hprev
                            sumC += this.Uc.data[j * hDim + k] * hprev
                            k++
                        }

                        let iGate = sigmoid(sumI)
                        let fGate = sigmoid(sumF)
                        let oGate = sigmoid(sumO)
                        let cTilde = fastTanh(sumC)

                        this.lastI[baseI + j] = iGate
                        this.lastF[baseF + j] = fGate
                        this.lastO[baseO + j] = oGate
                        this.lastCtilde[baseCtilde + j] = cTilde

                        // Cell state
                        let cprev = this.lastC[baseCprev + j]
                        let ccur = fGate * cprev + iGate * cTilde
                        this.lastC[baseCcur + j] = ccur

                        // Hidden state
                        let hcur = oGate * fastTanh(ccur)
                        this.lastH[baseHcur + j] = hcur

                        outData[(b * seq + t) * hDim + j] = hcur

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
            while (i < this.dWi.data.length) { this.dWi.data[i] = 0; i++ }
            i = 0
            while (i < this.dUi.data.length) { this.dUi.data[i] = 0; i++ }
            i = 0
            while (i < this.dBi.data.length) { this.dBi.data[i] = 0; i++ }

            i = 0
            while (i < this.dWf.data.length) { this.dWf.data[i] = 0; i++ }
            i = 0
            while (i < this.dUf.data.length) { this.dUf.data[i] = 0; i++ }
            i = 0
            while (i < this.dBf.data.length) { this.dBf.data[i] = 0; i++ }

            i = 0
            while (i < this.dWo.data.length) { this.dWo.data[i] = 0; i++ }
            i = 0
            while (i < this.dUo.data.length) { this.dUo.data[i] = 0; i++ }
            i = 0
            while (i < this.dBo.data.length) { this.dBo.data[i] = 0; i++ }

            i = 0
            while (i < this.dWc.data.length) { this.dWc.data[i] = 0; i++ }
            i = 0
            while (i < this.dUc.data.length) { this.dUc.data[i] = 0; i++ }
            i = 0
            while (i < this.dBc.data.length) { this.dBc.data[i] = 0; i++ }

            let dXdata = alloc(x.data.length)
            let dHnext = alloc(batch * hDim)
            let dCnext = alloc(batch * hDim)

            let t = seq - 1
            while (t >= 0) {
                let b = 0
                while (b < batch) {

                    let baseX = (b * seq + t) * inDim
                    let baseHprev = (b * (seq + 1) + t) * hDim
                    let baseCprev = (b * (seq + 1) + t) * hDim
                    let baseHcur = (b * (seq + 1) + (t + 1)) * hDim
                    let baseCcur = (b * (seq + 1) + (t + 1)) * hDim

                    let baseI = (b * seq + t) * hDim
                    let baseF = (b * seq + t) * hDim
                    let baseO = (b * seq + t) * hDim
                    let baseCtilde = (b * seq + t) * hDim

                    // dL/dh_t
                    let dH = alloc(hDim)
                    let j = 0
                    while (j < hDim) {
                        dH[j] = gradOut.data[(b * seq + t) * hDim + j] + dHnext[b * hDim + j]
                        j++
                    }

                    // dL/dc_t
                    let dC = alloc(hDim)
                    j = 0
                    while (j < hDim) {
                        let o = this.lastO[baseO + j]
                        let ccur = this.lastC[baseCcur + j]
                        dC[j] = dH[j] * o * (1 - fastTanh(ccur) * fastTanh(ccur)) + dCnext[b * hDim + j]
                        j++
                    }

                    // Gate derivatives
                    let dI = alloc(hDim)
                    let dF = alloc(hDim)
                    let dO = alloc(hDim)
                    let dCtilde = alloc(hDim)

                    j = 0
                    while (j < hDim) {
                        let iGate = this.lastI[baseI + j]
                        let fGate = this.lastF[baseF + j]
                        let oGate = this.lastO[baseO + j]
                        let cTilde = this.lastCtilde[baseCtilde + j]
                        let cprev = this.lastC[baseCprev + j]

                        dI[j] = dC[j] * cTilde * iGate * (1 - iGate)
                        dF[j] = dC[j] * cprev * fGate * (1 - fGate)
                        dO[j] = dH[j] * fastTanh(this.lastC[baseCcur + j]) * oGate * (1 - oGate)
                        dCtilde[j] = dC[j] * iGate * (1 - cTilde * cTilde)

                        j++
                    }

                    // ---- Accumulate gradients ----

                    // dWi, dUi, dBi
                    j = 0
                    while (j < hDim) {
                        this.dBi.data[j] += dI[j]

                        let k = 0
                        while (k < inDim) {
                            this.dWi.data[j * inDim + k] += dI[j] * x.data[baseX + k]
                            k++
                        }

                        k = 0
                        while (k < hDim) {
                            this.dUi.data[j * hDim + k] += dI[j] * this.lastH[baseHprev + k]
                            k++
                        }

                        j++
                    }

                    // dWf, dUf, dBf
                    j = 0
                    while (j < hDim) {
                        this.dBf.data[j] += dF[j]

                        let k = 0
                        while (k < inDim) {
                            this.dWf.data[j * inDim + k] += dF[j] * x.data[baseX + k]
                            k++
                        }

                        k = 0
                        while (k < hDim) {
                            this.dUf.data[j * hDim + k] += dF[j] * this.lastH[baseHprev + k]
                            k++
                        }

                        j++
                    }

                    // dWo, dUo, dBo
                    j = 0
                    while (j < hDim) {
                        this.dBo.data[j] += dO[j]

                        let k = 0
                        while (k < inDim) {
                            this.dWo.data[j * inDim + k] += dO[j] * x.data[baseX + k]
                            k++
                        }

                        k = 0
                        while (k < hDim) {
                            this.dUo.data[j * hDim + k] += dO[j] * this.lastH[baseHprev + k]
                            k++
                        }

                        j++
                    }

                    // dWc, dUc, dBc
                    j = 0
                    while (j < hDim) {
                        this.dBc.data[j] += dCtilde[j]

                        let k = 0
                        while (k < inDim) {
                            this.dWc.data[j * inDim + k] += dCtilde[j] * x.data[baseX + k]
                            k++
                        }

                        k = 0
                        while (k < hDim) {
                            this.dUc.data[j * hDim + k] += dCtilde[j] * this.lastH[baseHprev + k]
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
                            sum += dI[k] * this.Wi.data[k * inDim + j]
                            sum += dF[k] * this.Wf.data[k * inDim + j]
                            sum += dO[k] * this.Wo.data[k * inDim + j]
                            sum += dCtilde[k] * this.Wc.data[k * inDim + j]
                            k++
                        }
                        dXdata[baseX + j] = sum
                        j++
                    }

                    // ---- Compute dHnext and dCnext ----
                    j = 0
                    while (j < hDim) {
                        // dCnext = dC * f_t
                        let fGate = this.lastF[baseF + j]
                        dCnext[b * hDim + j] = dC[j] * fGate

                        // dHnext = contributions from all gates
                        let sumH = 0

                        // From input gate
                        let k = 0
                        while (k < hDim) {
                            sumH += dI[k] * this.Ui.data[k * hDim + j]
                            k++
                        }

                        // From forget gate
                        k = 0
                        while (k < hDim) {
                            sumH += dF[k] * this.Uf.data[k * hDim + j]
                            k++
                        }

                        // From output gate
                        k = 0
                        while (k < hDim) {
                            sumH += dO[k] * this.Uo.data[k * hDim + j]
                            k++
                        }

                        // From candidate cell
                        k = 0
                        while (k < hDim) {
                            sumH += dCtilde[k] * this.Uc.data[k * hDim + j]
                            k++
                        }

                        dHnext[b * hDim + j] = sumH

                        j++
                    }

                    b++
                    t--
                }
            }

            return new Tensor(dXdata, x.shape.slice(0))
        }
    }

}