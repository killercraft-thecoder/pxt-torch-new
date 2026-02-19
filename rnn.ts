namespace TorchNew {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0, size)
        return arr
    }


    // tanh implementation (same as in activations.ts)
    function fastTanh(x: number): number {
        let e2x = Math.exp(2 * x)
        return (e2x - 1) / (e2x + 1)
    }

    export class RNN {
        inputDim: number
        hiddenDim: number

        Wxh: Tensor
        Whh: Tensor
        b: Tensor

        dWxh: Tensor
        dWhh: Tensor
        dB: Tensor

        lastInput: Tensor
        lastH: number[]      // all hidden states (flattened)
        lastA: number[]      // all pre-activations (Wxh x + Whh h + b)

        constructor(inputDim: number, hiddenDim: number) {
            this.inputDim = inputDim
            this.hiddenDim = hiddenDim

            // Weight shapes
            this.Wxh = new Tensor(
                this.initWeights(inputDim * hiddenDim),
                [hiddenDim, inputDim]
            )

            this.Whh = new Tensor(
                this.initWeights(hiddenDim * hiddenDim),
                [hiddenDim, hiddenDim]
            )

            this.b = new Tensor(
                alloc(hiddenDim),
                [hiddenDim]
            )

            this.dWxh = new Tensor(alloc(inputDim * hiddenDim), [hiddenDim, inputDim])
            this.dWhh = new Tensor(alloc(hiddenDim * hiddenDim), [hiddenDim, hiddenDim])
            this.dB = new Tensor(alloc(hiddenDim), [hiddenDim])
        }

        initWeights(size: number): number[] {
            let arr = alloc(size)
            let scale = 1 / Math.sqrt(size)
            let i = 0
            while (i < size) {
                arr[i] = (Math.random() - 0.5) * 2 * scale
                i++
            }
            return arr
        }

        // ---------------------------------------------------------
        // Forward
        // ---------------------------------------------------------
        forward(x: Tensor): Tensor {
            // x shape: [batch, seq, inputDim]
            this.lastInput = x

            let batch = x.shape[0]
            let seq = x.shape[1]
            let inDim = this.inputDim
            let hDim = this.hiddenDim

            let outData = alloc(batch * seq * hDim)

            // Store hidden states and pre-activations
            this.lastH = alloc(batch * (seq + 1) * hDim)  // includes h0 = 0
            this.lastA = alloc(batch * seq * hDim)

            // Initialize h0 = 0
            let i = 0
            while (i < batch * hDim) {
                this.lastH[i] = 0
                i++
            }

            // Forward through time
            let t = 0
            while (t < seq) {
                let b = 0
                while (b < batch) {

                    // Compute h_t
                    let baseX = (b * seq + t) * inDim
                    let baseHprev = (b * (seq + 1) + t) * hDim
                    let baseHcur = (b * (seq + 1) + (t + 1)) * hDim
                    let baseA = (b * seq + t) * hDim

                    let j = 0
                    while (j < hDim) {
                        let sum = this.b.data[j]

                        // Wxh * x_t
                        let k = 0
                        while (k < inDim) {
                            let w = this.Wxh.data[j * inDim + k]
                            let xv = x.data[baseX + k]
                            sum += w * xv
                            k++
                        }

                        // Whh * h_{t-1}
                        k = 0
                        while (k < hDim) {
                            let w = this.Whh.data[j * hDim + k]
                            let hv = this.lastH[baseHprev + k]
                            sum += w * hv
                            k++
                        }

                        this.lastA[baseA + j] = sum
                        this.lastH[baseHcur + j] = fastTanh(sum)

                        // Output is h_t
                        outData[(b * seq + t) * hDim + j] = this.lastH[baseHcur + j]

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
            while (i < this.dWxh.data.length) { this.dWxh.data[i] = 0; i++ }
            i = 0
            while (i < this.dWhh.data.length) { this.dWhh.data[i] = 0; i++ }
            i = 0
            while (i < this.dB.data.length) { this.dB.data[i] = 0; i++ }

            let dXdata = alloc(x.data.length)
            let dHnext = alloc(batch * hDim)  // gradient from next time step

            // BPTT
            let t = seq - 1
            while (t >= 0) {
                let b = 0
                while (b < batch) {

                    let baseX = (b * seq + t) * inDim
                    let baseHprev = (b * (seq + 1) + t) * hDim
                    let baseHcur = (b * (seq + 1) + (t + 1)) * hDim
                    let baseA = (b * seq + t) * hDim

                    // dL/dh_t = gradOut + dHnext
                    let dH = alloc(hDim)
                    let j = 0
                    while (j < hDim) {
                        dH[j] = gradOut.data[(b * seq + t) * hDim + j] + dHnext[b * hDim + j]
                        j++
                    }

                    // dA = dH * (1 - tanh(a)^2)
                    let dA = alloc(hDim)
                    j = 0
                    while (j < hDim) {
                        let h = this.lastH[baseHcur + j]
                        dA[j] = dH[j] * (1 - h * h)
                        j++
                    }

                    // Accumulate gradients
                    j = 0
                    while (j < hDim) {
                        // dB
                        this.dB.data[j] += dA[j]

                        // dWxh
                        let k = 0
                        while (k < inDim) {
                            let idx = j * inDim + k
                            this.dWxh.data[idx] += dA[j] * x.data[baseX + k]
                            k++
                        }

                        // dWhh
                        k = 0
                        while (k < hDim) {
                            let idx = j * hDim + k
                            this.dWhh.data[idx] += dA[j] * this.lastH[baseHprev + k]
                            k++
                        }

                        j++
                    }

                    // dX
                    j = 0
                    while (j < inDim) {
                        let sum = 0
                        let k = 0
                        while (k < hDim) {
                            let w = this.Wxh.data[k * inDim + j]
                            sum += dA[k] * w
                            k++
                        }
                        dXdata[baseX + j] = sum
                        j++
                    }

                    // dHnext = Whh^T * dA
                    j = 0
                    while (j < hDim) {
                        let sum = 0
                        let k = 0
                        while (k < hDim) {
                            let w = this.Whh.data[k * hDim + j]
                            sum += dA[k] * w
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