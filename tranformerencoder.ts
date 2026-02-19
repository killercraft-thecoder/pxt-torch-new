namespace TorchNew {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0, size)
        return arr
    }

    export class TransformerEncoder {
        ln1: LayerNorm
        ln2: LayerNorm
        mha: MultiHeadAttention
        ff: FeedForward

        lastInput: Tensor
        lastAfterMHA: Tensor
        lastAfterFF: Tensor

        constructor(embedDim: number, numHeads: number, ffHiddenDim: number) {
            // LayerNorm expects an array of dims
            this.ln1 = new LayerNorm([embedDim])
            this.mha = new MultiHeadAttention(embedDim, numHeads)

            this.ln2 = new LayerNorm([embedDim])

            // FeedForward expects activation instance
            this.ff = new FeedForward(
                embedDim,
                ffHiddenDim,
                embedDim,
                new TorchNew.Activations.ReLU()
            )
        }

        // ---------------------------------------------------------
        // Forward
        // ---------------------------------------------------------
        forward(x: Tensor): Tensor {
            this.lastInput = x

            // 1) LayerNorm → MHA → Residual
            let xNorm1 = this.ln1.forward(x)
            let mhaOut = this.mha.forward(xNorm1)

            let size = x.data.length
            let afterMHAdata = alloc(size)
            let i = 0
            while (i < size) {
                afterMHAdata[i] = x.data[i] + mhaOut.data[i]
                i++
            }
            let afterMHA = new Tensor(afterMHAdata, x.shape.slice(0))
            this.lastAfterMHA = afterMHA

            // 2) LayerNorm → FeedForward → Residual
            let xNorm2 = this.ln2.forward(afterMHA)
            let ffOut = this.ff.forward(xNorm2)

            let afterFFdata = alloc(size)
            i = 0
            while (i < size) {
                afterFFdata[i] = afterMHA.data[i] + ffOut.data[i]
                i++
            }
            let afterFF = new Tensor(afterFFdata, x.shape.slice(0))
            this.lastAfterFF = afterFF

            return afterFF
        }

        // ---------------------------------------------------------
        // Backward
        // ---------------------------------------------------------
        backward(gradOut: Tensor): Tensor {
            let size = gradOut.data.length

            // 1) Backprop through second residual: d(afterMHA) += gradOut
            let dAfterMHAdata = alloc(size)
            let dFFout = alloc(size)

            let i = 0
            while (i < size) {
                dAfterMHAdata[i] = gradOut.data[i]
                dFFout[i] = gradOut.data[i]
                i++
            }

            // 2) Backprop through FeedForward
            let dFF = this.ff.backward(new Tensor(dFFout, gradOut.shape.slice(0)))

            // 3) Backprop through LayerNorm2
            let dNorm2 = this.ln2.backward(dFF)

            // Add to dAfterMHA
            i = 0
            while (i < size) {
                dAfterMHAdata[i] += dNorm2.data[i]
                i++
            }

            let dAfterMHA = new Tensor(dAfterMHAdata, gradOut.shape.slice(0))

            // 4) Backprop through first residual: dX += dAfterMHA
            let dXdata = alloc(size)
            let dMHAout = alloc(size)

            i = 0
            while (i < size) {
                dXdata[i] = dAfterMHA.data[i]
                dMHAout[i] = dAfterMHA.data[i]
                i++
            }

            // 5) Backprop through MHA
            let dMHA = this.mha.backward(new Tensor(dMHAout, gradOut.shape.slice(0)))

            // 6) Backprop through LayerNorm1
            let dNorm1 = this.ln1.backward(dMHA)

            // Add to dX
            i = 0
            while (i < size) {
                dXdata[i] += dNorm1.data[i]
                i++
            }

            return new Tensor(dXdata, gradOut.shape.slice(0))
        }
    }
}