namespace TorchNew {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0, size)
        return arr
    }

    export class TransformerDecoder {
        ln1: LayerNorm
        ln2: LayerNorm
        ln3: LayerNorm

        selfAtt: MultiHeadAttention
        crossAtt: MultiHeadAttention
        ff: FeedForward

        lastInput: Tensor
        lastAfterSelf: Tensor
        lastAfterCross: Tensor
        lastAfterFF: Tensor

        lastDEnc: Tensor



        constructor(embedDim: number, numHeads: number, ffHiddenDim: number) {
            this.ln1 = new LayerNorm([embedDim])
            this.selfAtt = new MultiHeadAttention(embedDim, numHeads)

            this.ln2 = new LayerNorm([embedDim])
            this.crossAtt = new MultiHeadAttention(embedDim, numHeads)

            this.ln3 = new LayerNorm([embedDim])
            this.ff = new FeedForward(embedDim, ffHiddenDim, embedDim, new TorchNew.Activations.ReLU())
        }

        // ---------------------------------------------------------
        // Forward
        // ---------------------------------------------------------
        forward(x: Tensor, encoderOut: Tensor): Tensor {
            this.lastInput = x

            let batch = x.shape[0]
            let seq = x.shape[1]
            let embedDim = x.shape[2]

            // -----------------------------
            // 1) Masked Self-Attention
            // -----------------------------
            let xNorm1 = this.ln1.forward(x)

            // Apply causal mask: disallow attending to future tokens
            let masked = this.applyCausalMask(xNorm1)

            let selfOut = this.selfAtt.forward(masked)

            let size = x.data.length
            let afterSelfData = alloc(size)
            let i = 0
            while (i < size) {
                afterSelfData[i] = x.data[i] + selfOut.data[i]
                i++
            }
            let afterSelf = new Tensor(afterSelfData, x.shape.slice(0))
            this.lastAfterSelf = afterSelf

            // -----------------------------
            // 2) Cross-Attention
            // -----------------------------
            let xNorm2 = this.ln2.forward(afterSelf)

            // Cross-attention: query = decoder, key/value = encoder
            let crossOut = this.crossAtt.forwardKV(xNorm2, encoderOut, encoderOut)

            let afterCrossData = alloc(size)
            i = 0
            while (i < size) {
                afterCrossData[i] = afterSelf.data[i] + crossOut.data[i]
                i++
            }
            let afterCross = new Tensor(afterCrossData, x.shape.slice(0))
            this.lastAfterCross = afterCross

            // -----------------------------
            // 3) FeedForward
            // -----------------------------
            let xNorm3 = this.ln3.forward(afterCross)
            let ffOut = this.ff.forward(xNorm3)

            let afterFFData = alloc(size)
            i = 0
            while (i < size) {
                afterFFData[i] = afterCross.data[i] + ffOut.data[i]
                i++
            }
            let afterFF = new Tensor(afterFFData, x.shape.slice(0))
            this.lastAfterFF = afterFF

            return afterFF
        }

        // ---------------------------------------------------------
        // Backward
        // ---------------------------------------------------------
        backward(gradOut: Tensor, encoderOut: Tensor): Tensor {
            let size = gradOut.data.length

            // -----------------------------
            // 1) Backprop through FF residual
            // -----------------------------
            let dAfterCrossData = alloc(size)
            let dFFout = alloc(size)

            let i = 0
            while (i < size) {
                dAfterCrossData[i] = gradOut.data[i]
                dFFout[i] = gradOut.data[i]
                i++
            }

            let dFF = this.ff.backward(new Tensor(dFFout, gradOut.shape.slice(0)))
            let dNorm3 = this.ln3.backward(dFF)

            i = 0
            while (i < size) {
                dAfterCrossData[i] += dNorm3.data[i]
                i++
            }

            let dAfterCross = new Tensor(dAfterCrossData, gradOut.shape.slice(0))

            // -----------------------------
            // 2) Backprop through cross-attention residual
            // -----------------------------
            let dAfterSelfData = alloc(size)
            let dCrossOut = alloc(size)

            i = 0
            while (i < size) {
                dAfterSelfData[i] = dAfterCross.data[i]
                dCrossOut[i] = dAfterCross.data[i]
                i++
            }

            let dCross = this.crossAtt.backwardKV(
                new Tensor(dCrossOut, gradOut.shape.slice(0)),
                encoderOut,
                encoderOut
            )

            let dEncData = alloc(encoderOut.data.length)
            let j = 0
            while (j < dEncData.length) {
                dEncData[j] = this.crossAtt.lastDK.data[j] + this.crossAtt.lastDV.data[j]
                j++
            }
            this.lastDEnc = new Tensor(dEncData, encoderOut.shape.slice(0))


            let dNorm2 = this.ln2.backward(dCross)

            i = 0
            while (i < size) {
                dAfterSelfData[i] += dNorm2.data[i]
                i++
            }

            let dAfterSelf = new Tensor(dAfterSelfData, gradOut.shape.slice(0))

            // -----------------------------
            // 3) Backprop through self-attention residual
            // -----------------------------
            let dXdata = alloc(size)
            let dSelfOut = alloc(size)

            i = 0
            while (i < size) {
                dXdata[i] = dAfterSelf.data[i]
                dSelfOut[i] = dAfterSelf.data[i]
                i++
            }

            let dSelf = this.selfAtt.backward(new Tensor(dSelfOut, gradOut.shape.slice(0)))
            let dNorm1 = this.ln1.backward(dSelf)

            i = 0
            while (i < size) {
                dXdata[i] += dNorm1.data[i]
                i++
            }

            return new Tensor(dXdata, gradOut.shape.slice(0))
        }

        // ---------------------------------------------------------
        // Causal mask: zero out future positions
        // ---------------------------------------------------------
        applyCausalMask(x: Tensor): Tensor {
            let batch = x.shape[0]
            let seq = x.shape[1]
            let dim = x.shape[2]

            let outData = alloc(x.data.length)

            let b = 0
            while (b < batch) {
                let i = 0
                while (i < seq) {
                    let j = 0
                    while (j < seq) {
                        let k = 0
                        while (k < dim) {
                            let idx = b * seq * dim + i * dim + k
                            let src = b * seq * dim + j * dim + k

                            if (j <= i) {
                                outData[idx] = x.data[src]
                            } else {
                                outData[idx] = 0
                            }

                            k++
                        }
                        j++
                    }
                    i++
                }
                b++
            }

            return new Tensor(outData, x.shape.slice(0))
        }
    }
}