namespace TorchNew {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0, size)
        return arr
    }

    function softmaxRow(data: number[], start: number, length: number): void {
        let maxVal = data[start]
        let i = 1
        while (i < length) {
            let v = data[start + i]
            if (v > maxVal) maxVal = v
            i++
        }

        let sum = 0
        i = 0
        while (i < length) {
            let e = Math.exp(data[start + i] - maxVal)
            data[start + i] = e
            sum += e
            i++
        }

        i = 0
        while (i < length) {
            data[start + i] /= sum
            i++
        }
    }

    export class MultiHeadAttention {
        embedDim: number
        numHeads: number
        headDim: number

        Wq: Tensor; bq: Tensor
        Wk: Tensor; bk: Tensor
        Wv: Tensor; bv: Tensor
        Wo: Tensor; bo: Tensor

        dWq: Tensor; dbq: Tensor
        dWk: Tensor; dbk: Tensor
        dWv: Tensor; dbv: Tensor
        dWo: Tensor; dbo: Tensor

        lastInput: Tensor
        lastQ: Tensor
        lastK: Tensor
        lastV: Tensor
        lastScores: number[]
        lastSoftmax: number[]
        lastAttention: number[]

        lastDK: Tensor
        lastDV: Tensor

        constructor(embedDim: number, numHeads: number) {
            this.embedDim = embedDim
            this.numHeads = numHeads
            this.headDim = Math.idiv(embedDim, numHeads)

            this.Wq = new Linear(embedDim, embedDim).weight
            this.bq = new Linear(embedDim, embedDim).bias

            this.Wk = new Linear(embedDim, embedDim).weight
            this.bk = new Linear(embedDim, embedDim).bias

            this.Wv = new Linear(embedDim, embedDim).weight
            this.bv = new Linear(embedDim, embedDim).bias

            this.Wo = new Linear(embedDim, embedDim).weight
            this.bo = new Linear(embedDim, embedDim).bias

            this.dWq = new Tensor(alloc(embedDim * embedDim), [embedDim, embedDim])
            this.dbq = new Tensor(alloc(embedDim), [embedDim])

            this.dWk = new Tensor(alloc(embedDim * embedDim), [embedDim, embedDim])
            this.dbk = new Tensor(alloc(embedDim), [embedDim])

            this.dWv = new Tensor(alloc(embedDim * embedDim), [embedDim, embedDim])
            this.dbv = new Tensor(alloc(embedDim), [embedDim])

            this.dWo = new Tensor(alloc(embedDim * embedDim), [embedDim, embedDim])
            this.dbo = new Tensor(alloc(embedDim), [embedDim])
        }

        // ---------------------------------------------------------
        // Forward
        // ---------------------------------------------------------
        forward(x: Tensor): Tensor {
            // x: [batch, seq, embedDim]
            this.lastInput = x

            let batch = x.shape[0]
            let seq = x.shape[1]
            let E = this.embedDim
            let H = this.numHeads
            let D = this.headDim

            // Compute Q, K, V
            let Q = this.linearForward(x, this.Wq, this.bq)
            let K = this.linearForward(x, this.Wk, this.bk)
            let V = this.linearForward(x, this.Wv, this.bv)

            this.lastQ = Q
            this.lastK = K
            this.lastV = V

            // Allocate scores and softmax
            let scores = alloc(batch * H * seq * seq)
            let softmaxOut = alloc(batch * H * seq * seq)
            let attention = alloc(batch * seq * E)

            this.lastScores = scores
            this.lastSoftmax = softmaxOut
            this.lastAttention = attention

            let scale = 1 / Math.sqrt(D)

            // Compute attention scores
            let b = 0
            while (b < batch) {
                let h = 0
                while (h < H) {
                    let qBase = (b * seq * E) + h * D
                    let kBase = (b * seq * E) + h * D

                    let sBase = (b * H * seq * seq) + (h * seq * seq)

                    let i = 0
                    while (i < seq) {
                        let j = 0
                        while (j < seq) {
                            let sum = 0
                            let d = 0
                            while (d < D) {
                                let qv = Q.data[qBase + i * E + d]
                                let kv = K.data[kBase + j * E + d]
                                sum += qv * kv
                                d++
                            }
                            scores[sBase + i * seq + j] = sum * scale
                            j++
                        }
                        i++
                    }

                    // Softmax row-wise
                    let i2 = 0
                    while (i2 < seq) {
                        softmaxRow(scores, sBase + i2 * seq, seq)
                        let j2 = 0
                        while (j2 < seq) {
                            softmaxOut[sBase + i2 * seq + j2] = scores[sBase + i2 * seq + j2]
                            j2++
                        }
                        i2++
                    }

                    // Weighted sum: attention = softmax * V
                    let outBase = b * seq * E
                    let i3 = 0
                    while (i3 < seq) {
                        let d3 = 0
                        while (d3 < D) {
                            let sum = 0
                            let j3 = 0
                            while (j3 < seq) {
                                let w = softmaxOut[sBase + i3 * seq + j3]
                                let vv = V.data[kBase + j3 * E + d3]
                                sum += w * vv
                                j3++
                            }
                            attention[outBase + i3 * E + h * D + d3] = sum
                            d3++
                        }
                        i3++
                    }

                    h++
                }
                b++
            }

            // Output projection
            let out = this.linearForward(
                new Tensor(attention, [batch, seq, E]),
                this.Wo,
                this.bo
            )

            return out
        }

        // ---------------------------------------------------------
        // Backward
        // ---------------------------------------------------------
        backward(gradOut: Tensor): Tensor {
            let x = this.lastInput
            let Q = this.lastQ
            let K = this.lastK
            let V = this.lastV

            let batch = x.shape[0]
            let seq = x.shape[1]
            let E = this.embedDim
            let H = this.numHeads
            let D = this.headDim

            // Zero parameter grads
            let i = 0
            while (i < this.dWq.data.length) { this.dWq.data[i] = 0; i++ }
            i = 0
            while (i < this.dbq.data.length) { this.dbq.data[i] = 0; i++ }
            i = 0
            while (i < this.dWk.data.length) { this.dWk.data[i] = 0; i++ }
            i = 0
            while (i < this.dbk.data.length) { this.dbk.data[i] = 0; i++ }
            i = 0
            while (i < this.dWv.data.length) { this.dWv.data[i] = 0; i++ }
            i = 0
            while (i < this.dbv.data.length) { this.dbv.data[i] = 0; i++ }
            i = 0
            while (i < this.dWo.data.length) { this.dWo.data[i] = 0; i++ }
            i = 0
            while (i < this.dbo.data.length) { this.dbo.data[i] = 0; i++ }

            // 1) Back through output projection: attention -> gradAtt
            let attTensor = new Tensor(this.lastAttention, [batch, seq, E])
            let gradAtt = this.linearBackward(
                attTensor,
                gradOut,
                this.Wo,
                this.dWo,
                this.dbo
            )

            // 2) Allocate grads for Q, K, V and softmax/scores
            let dQdata = alloc(batch * seq * E)
            let dKdata = alloc(batch * seq * E)
            let dVdata = alloc(batch * seq * E)

            let dSoft = alloc(batch * H * seq * seq)
            let dScores = alloc(batch * H * seq * seq)

            let scale = 1 / Math.sqrt(D)

            // 3) Backprop through attention = softmax * V
            let b = 0
            while (b < batch) {
                let h = 0
                while (h < H) {
                    let sBase = (b * H * seq * seq) + (h * seq * seq)
                    let outBase = b * seq * E
                    let qBase = (b * seq * E) + h * D
                    let kBase = (b * seq * E) + h * D
                    let vBase = (b * seq * E) + h * D

                    // dSoft and dV from gradAtt
                    let i2 = 0
                    while (i2 < seq) {
                        let d3 = 0
                        while (d3 < D) {
                            let g = gradAtt.data[outBase + i2 * E + h * D + d3]

                            let j2 = 0
                            while (j2 < seq) {
                                let w = this.lastSoftmax[sBase + i2 * seq + j2]
                                let vv = V.data[vBase + j2 * E + d3]

                                dSoft[sBase + i2 * seq + j2] += g * vv
                                dVdata[vBase + j2 * E + d3] += g * w

                                j2++
                            }

                            d3++
                        }
                        i2++
                    }

                    // 4) Softmax backward: dScores from dSoft and probs
                    let i3 = 0
                    while (i3 < seq) {
                        let dot = 0
                        let j3 = 0
                        while (j3 < seq) {
                            let p = this.lastSoftmax[sBase + i3 * seq + j3]
                            let g = dSoft[sBase + i3 * seq + j3]
                            dot += g * p
                            j3++
                        }

                        j3 = 0
                        while (j3 < seq) {
                            let p = this.lastSoftmax[sBase + i3 * seq + j3]
                            let g = dSoft[sBase + i3 * seq + j3]
                            dScores[sBase + i3 * seq + j3] = (g - dot) * p
                            j3++
                        }

                        i3++
                    }

                    // 5) Scores = (QK^T)/sqrt(D) -> dQ, dK
                    let i4 = 0
                    while (i4 < seq) {
                        let j4 = 0
                        while (j4 < seq) {
                            let ds = dScores[sBase + i4 * seq + j4] * scale

                            let d = 0
                            while (d < D) {
                                let qv = Q.data[qBase + i4 * E + d]
                                let kv = K.data[kBase + j4 * E + d]

                                dQdata[qBase + i4 * E + d] += ds * kv
                                dKdata[kBase + j4 * E + d] += ds * qv

                                d++
                            }

                            j4++
                        }
                        i4++
                    }

                    h++
                }
                b++
            }

            // 6) Merge dQ, dK, dV back to [batch, seq, E] tensors
            let dQ = new Tensor(dQdata, [batch, seq, E])
            let dK = new Tensor(dKdata, [batch, seq, E])
            let dV = new Tensor(dVdata, [batch, seq, E])

            // 7) Back through Q, K, V linears to input x
            let dXq = this.linearBackward(x, dQ, this.Wq, this.dWq, this.dbq)
            let dXk = this.linearBackward(x, dK, this.Wk, this.dWk, this.dbk)
            let dXv = this.linearBackward(x, dV, this.Wv, this.dWv, this.dbv)

            // 8) Sum contributions: dX = dXq + dXk + dXv
            let dXdata = alloc(x.data.length)
            i = 0
            while (i < dXdata.length) {
                dXdata[i] = dXq.data[i] + dXk.data[i] + dXv.data[i]
                i++
            }

            return new Tensor(dXdata, x.shape.slice(0))
        }

        // ---------------------------------------------------------
        // Helper: Linear forward
        // ---------------------------------------------------------
        linearForward(x: Tensor, W: Tensor, b: Tensor): Tensor {
            let batch = x.shape[0]
            let seq = x.shape[1]
            let inDim = x.shape[2]
            let outDim = b.shape[0]

            let outData = alloc(batch * seq * outDim)

            let bIdx = 0
            while (bIdx < batch) {
                let t = 0
                while (t < seq) {
                    let baseX = (bIdx * seq + t) * inDim
                    let baseO = (bIdx * seq + t) * outDim

                    let j = 0
                    while (j < outDim) {
                        let sum = b.data[j]

                        let k = 0
                        while (k < inDim) {
                            sum += W.data[j * inDim + k] * x.data[baseX + k]
                            k++
                        }

                        outData[baseO + j] = sum
                        j++
                    }

                    t++
                }
                bIdx++
            }

            return new Tensor(outData, [batch, seq, outDim])
        }

        // ---------------------------------------------------------
        // Helper: Linear backward (accumulates dW, db)
        // ---------------------------------------------------------
        linearBackward(x: Tensor, gradOut: Tensor, W: Tensor, dW: Tensor, db: Tensor): Tensor {
            let batch = x.shape[0]
            let seq = x.shape[1]
            let inDim = x.shape[2]
            let outDim = gradOut.shape[2]

            let gradInData = alloc(batch * seq * inDim)

            let bIdx = 0
            while (bIdx < batch) {
                let t = 0
                while (t < seq) {
                    let baseX = (bIdx * seq + t) * inDim
                    let baseG = (bIdx * seq + t) * outDim

                    let j = 0
                    while (j < outDim) {
                        let g = gradOut.data[baseG + j]

                        db.data[j] += g

                        let k = 0
                        while (k < inDim) {
                            dW.data[j * inDim + k] += g * x.data[baseX + k]
                            gradInData[baseX + k] += g * W.data[j * inDim + k]
                            k++
                        }

                        j++
                    }

                    t++
                }
                bIdx++
            }

            return new Tensor(gradInData, [batch, seq, inDim])
        }

        forwardKV(qInput: Tensor, kInput: Tensor, vInput: Tensor): Tensor {
            this.lastInput = qInput

            let batch = qInput.shape[0]
            let qSeq = qInput.shape[1]
            let kvSeq = kInput.shape[1]
            let E = this.embedDim
            let H = this.numHeads
            let D = this.headDim

            // Compute Q, K, V
            let Q = this.linearForward(qInput, this.Wq, this.bq)
            let K = this.linearForward(kInput, this.Wk, this.bk)
            let V = this.linearForward(vInput, this.Wv, this.bv)

            this.lastQ = Q
            this.lastK = K
            this.lastV = V

            // Allocate scores, softmax, attention
            let scores = alloc(batch * H * qSeq * kvSeq)
            let softmaxOut = alloc(batch * H * qSeq * kvSeq)
            let attention = alloc(batch * qSeq * E)

            this.lastScores = scores
            this.lastSoftmax = softmaxOut
            this.lastAttention = attention

            let scale = 1 / Math.sqrt(D)

            let b = 0
            while (b < batch) {
                let h = 0
                while (h < H) {
                    let qBase = (b * qSeq * E) + h * D
                    let kBase = (b * kvSeq * E) + h * D
                    let vBase = (b * kvSeq * E) + h * D

                    let sBase = (b * H * qSeq * kvSeq) + (h * qSeq * kvSeq)

                    // Compute scores = QK^T / sqrt(D)
                    let i = 0
                    while (i < qSeq) {
                        let j = 0
                        while (j < kvSeq) {
                            let sum = 0
                            let d = 0
                            while (d < D) {
                                let qv = Q.data[qBase + i * E + d]
                                let kv = K.data[kBase + j * E + d]
                                sum += qv * kv
                                d++
                            }
                            scores[sBase + i * kvSeq + j] = sum * scale
                            j++
                        }
                        i++
                    }

                    // Softmax row-wise
                    let i2 = 0
                    while (i2 < qSeq) {
                        softmaxRow(scores, sBase + i2 * kvSeq, kvSeq)
                        let j2 = 0
                        while (j2 < kvSeq) {
                            softmaxOut[sBase + i2 * kvSeq + j2] = scores[sBase + i2 * kvSeq + j2]
                            j2++
                        }
                        i2++
                    }

                    // Weighted sum: attention = softmax * V
                    let outBase = b * qSeq * E
                    let i3 = 0
                    while (i3 < qSeq) {
                        let d3 = 0
                        while (d3 < D) {
                            let sum = 0
                            let j3 = 0
                            while (j3 < kvSeq) {
                                let w = softmaxOut[sBase + i3 * kvSeq + j3]
                                let vv = V.data[vBase + j3 * E + d3]
                                sum += w * vv
                                j3++
                            }
                            attention[outBase + i3 * E + h * D + d3] = sum
                            d3++
                        }
                        i3++
                    }

                    h++
                }
                b++
            }

            // Output projection
            let out = this.linearForward(
                new Tensor(attention, [batch, qSeq, E]),
                this.Wo,
                this.bo
            )

            return out
        }

        backwardKV(gradOut: Tensor, kInput: Tensor, vInput: Tensor): Tensor {
            let Q = this.lastQ
            let K = this.lastK
            let V = this.lastV

            let batch = Q.shape[0]
            let qSeq = Q.shape[1]
            let kvSeq = K.shape[1]
            let E = this.embedDim
            let H = this.numHeads
            let D = this.headDim

            // Zero parameter grads
            let i = 0
            while (i < this.dWq.data.length) { this.dWq.data[i] = 0; i++ }
            i = 0
            while (i < this.dbq.data.length) { this.dbq.data[i] = 0; i++ }
            i = 0
            while (i < this.dWk.data.length) { this.dWk.data[i] = 0; i++ }
            i = 0
            while (i < this.dbk.data.length) { this.dbk.data[i] = 0; i++ }
            i = 0
            while (i < this.dWv.data.length) { this.dWv.data[i] = 0; i++ }
            i = 0
            while (i < this.dbv.data.length) { this.dbv.data[i] = 0; i++ }
            i = 0
            while (i < this.dWo.data.length) { this.dWo.data[i] = 0; i++ }
            i = 0
            while (i < this.dbo.data.length) { this.dbo.data[i] = 0; i++ }

            // 1) Back through output projection
            let attTensor = new Tensor(this.lastAttention, [batch, qSeq, E])
            let gradAtt = this.linearBackward(
                attTensor,
                gradOut,
                this.Wo,
                this.dWo,
                this.dbo
            )

            // Allocate grads for Q, K, V
            let dQdata = alloc(batch * qSeq * E)
            let dKdata = alloc(batch * kvSeq * E)
            let dVdata = alloc(batch * kvSeq * E)

            let dSoft = alloc(batch * H * qSeq * kvSeq)
            let dScores = alloc(batch * H * qSeq * kvSeq)

            let scale = 1 / Math.sqrt(D)

            // 2) Backprop through attention = softmax * V
            let b = 0
            while (b < batch) {
                let h = 0
                while (h < H) {
                    let sBase = (b * H * qSeq * kvSeq) + (h * qSeq * kvSeq)
                    let outBase = b * qSeq * E
                    let qBase = (b * qSeq * E) + h * D
                    let kBase = (b * kvSeq * E) + h * D
                    let vBase = (b * kvSeq * E) + h * D

                    // dSoft and dV
                    let i2 = 0
                    while (i2 < qSeq) {
                        let d3 = 0
                        while (d3 < D) {
                            let g = gradAtt.data[outBase + i2 * E + h * D + d3]

                            let j2 = 0
                            while (j2 < kvSeq) {
                                let w = this.lastSoftmax[sBase + i2 * kvSeq + j2]
                                let vv = V.data[vBase + j2 * E + d3]

                                dSoft[sBase + i2 * kvSeq + j2] += g * vv
                                dVdata[vBase + j2 * E + d3] += g * w

                                j2++
                            }

                            d3++
                        }
                        i2++
                    }

                    // 3) Softmax backward
                    let i3 = 0
                    while (i3 < qSeq) {
                        let dot = 0
                        let j3 = 0
                        while (j3 < kvSeq) {
                            let p = this.lastSoftmax[sBase + i3 * kvSeq + j3]
                            let g = dSoft[sBase + i3 * kvSeq + j3]
                            dot += g * p
                            j3++
                        }

                        j3 = 0
                        while (j3 < kvSeq) {
                            let p = this.lastSoftmax[sBase + i3 * kvSeq + j3]
                            let g = dSoft[sBase + i3 * kvSeq + j3]
                            dScores[sBase + i3 * kvSeq + j3] = (g - dot) * p
                            j3++
                        }

                        i3++
                    }

                    // 4) Scores = QK^T / sqrt(D)
                    let i4 = 0
                    while (i4 < qSeq) {
                        let j4 = 0
                        while (j4 < kvSeq) {
                            let ds = dScores[sBase + i4 * kvSeq + j4] * scale

                            let d = 0
                            while (d < D) {
                                let qv = Q.data[qBase + i4 * E + d]
                                let kv = K.data[kBase + j4 * E + d]

                                dQdata[qBase + i4 * E + d] += ds * kv
                                dKdata[kBase + j4 * E + d] += ds * qv

                                d++
                            }

                            j4++
                        }
                        i4++
                    }

                    h++
                }
                b++
            }

            // 5) Wrap dQ, dK, dV as tensors
            let dQ = new Tensor(dQdata, [batch, qSeq, E])
            let dK = new Tensor(dKdata, [batch, kvSeq, E])
            let dV = new Tensor(dVdata, [batch, kvSeq, E])

            this.lastDK = dK
            this.lastDV = dV


            // 6) Back through Q, K, V linears
            let dXq = this.linearBackward(this.lastInput, dQ, this.Wq, this.dWq, this.dbq)
            let dXk = this.linearBackward(kInput, dK, this.Wk, this.dWk, this.dbk)
            let dXv = this.linearBackward(vInput, dV, this.Wv, this.dWv, this.dbv)

            // 7) Return gradients for decoder input (queries)
            return dXq
        }
    }
}