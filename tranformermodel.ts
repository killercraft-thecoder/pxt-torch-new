namespace TorchNew {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0, size)
        return arr
    }

    export class TransformerModel {
        embed: Embedding
        posEnc: PositionalEncoding

        encoderBlocks: TransformerEncoder[]
        decoderBlocks: TransformerDecoder[]

        finalLinear: Linear

        lastInput: Tensor
        lastEncoderOut: Tensor
        lastDecoderOut: Tensor

        constructor(
            vocabSize: number,
            embedDim: number,
            numHeads: number,
            ffHiddenDim: number,
            numEncoderLayers: number,
            numDecoderLayers: number,
            maxSeqLen: number
        ) {
            this.embed = new Embedding(vocabSize, embedDim)
            this.posEnc = new PositionalEncoding(maxSeqLen, embedDim)

            this.encoderBlocks = []
            let i = 0
            while (i < numEncoderLayers) {
                this.encoderBlocks.push(
                    new TransformerEncoder(embedDim, numHeads, ffHiddenDim)
                )
                i++
            }

            this.decoderBlocks = []
            i = 0
            while (i < numDecoderLayers) {
                this.decoderBlocks.push(
                    new TransformerDecoder(embedDim, numHeads, ffHiddenDim)
                )
                i++
            }

            this.finalLinear = new Linear(embedDim, vocabSize)
        }

        // ---------------------------------------------------------
        // Forward
        // ---------------------------------------------------------
        forward(srcTokens: Tensor, tgtTokens: Tensor): Tensor {
            // srcTokens: [batch, srcSeq]
            // tgtTokens: [batch, tgtSeq]

            this.lastInput = srcTokens

            // 1) Embed + positional encode source
            let srcEmb = this.embed.forward(srcTokens)
            let srcPE = this.posEnc.forward(srcEmb)

            // 2) Pass through encoder stack
            let encOut = srcPE
            let i = 0
            while (i < this.encoderBlocks.length) {
                encOut = this.encoderBlocks[i].forward(encOut)
                i++
            }
            this.lastEncoderOut = encOut

            // 3) Embed + positional encode target
            let tgtEmb = this.embed.forward(tgtTokens)
            let tgtPE = this.posEnc.forward(tgtEmb)

            // 4) Pass through decoder stack (with cross-attention)
            let decOut = tgtPE
            i = 0
            while (i < this.decoderBlocks.length) {
                decOut = this.decoderBlocks[i].forward(decOut, encOut)
                i++
            }
            this.lastDecoderOut = decOut

            // 5) Final projection to vocab logits
            let logits = this.finalLinear.forward(decOut)
            return logits
        }

        backward(gradOut: Tensor): { dSrc: Tensor, dTgt: Tensor } {
            // 1) Backprop through final linear projection
            // Use the same input x that was used in forward (adjust name if different)
            const linGrad = this.finalLinear.backward(gradOut, this.lastDecoderOut)
            let dDec: Tensor = linGrad.dx  // [batch, tgtSeq, embedDim]

            // 2) Backprop through decoder stack (right-to-left)
            let i = this.decoderBlocks.length - 1

            // Accumulate gradients w.r.t. encoder output from all decoder blocks
            let dEncAccum: Tensor = null

            while (i >= 0) {
                const block = this.decoderBlocks[i]

                // block.backward returns { dX, dEnc }
                // block.backward returns only dX (Tensor)
                const dDecIn: Tensor = block.backward(dDec, this.lastEncoderOut)

                // encoder gradient is stored internally
                const dEncFromBlock: Tensor = block.lastDEnc




                if (dEncAccum == null) {
                    dEncAccum = dEncFromBlock
                } else {
                    const size = dEncAccum.data.length
                    let j = 0
                    while (j < size) {
                        dEncAccum.data[j] += dEncFromBlock.data[j]
                        j++
                    }
                }

                dDec = dDecIn
                i--
            }

            // 3) Backprop through encoder stack (right-to-left)
            let dEnc: Tensor = dEncAccum
            i = this.encoderBlocks.length - 1
            while (i >= 0) {
                dEnc = this.encoderBlocks[i].backward(dEnc)
                i--
            }

            // 4) Backprop through positional encoding (source path)
            const dSrcPE: Tensor = this.posEnc.backward(dEnc)

            // 5) Backprop through embedding (source tokens)
            const dSrc: Tensor = this.embed.backward(dSrcPE)

            // 6) Backprop through positional encoding (target path)
            const dTgtPE: Tensor = this.posEnc.backward(dDec)

            // 7) Backprop through embedding (target tokens)
            const dTgt: Tensor = this.embed.backward(dTgtPE)

            return {
                dSrc: dSrc,
                dTgt: dTgt
            }
        }
    }
}