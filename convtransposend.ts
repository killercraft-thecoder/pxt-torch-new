namespace TorchNew {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0, size)
        return arr
    }

    export class ConvTransposeND {
        inChannels: number
        outChannels: number
        kernelShape: number[]
        stride: number[]
        padding: number[]

        weight: Tensor
        bias: Tensor

        dW: Tensor
        dB: Tensor

        lastInput: Tensor

        constructor(inChannels: number, outChannels: number,
            kernelShape: number[], stride: number[], padding: number[]) {

            this.inChannels = inChannels
            this.outChannels = outChannels
            this.kernelShape = kernelShape.slice(0)
            this.stride = stride.slice(0)
            this.padding = padding.slice(0)

            let dims = kernelShape.length

            // Weight shape: [inChannels, outChannels, ...kernelShape]
            let wShape = alloc(2 + dims)
            wShape[0] = inChannels
            wShape[1] = outChannels
            let i = 0
            while (i < dims) {
                wShape[2 + i] = kernelShape[i]
                i++
            }

            let wSize = 1
            i = 0
            while (i < wShape.length) {
                wSize *= wShape[i]
                i++
            }

            let wData = alloc(wSize)
            let scale = 1 / Math.sqrt(inChannels)
            i = 0
            while (i < wSize) {
                wData[i] = (Math.random() - 0.5) * 2 * scale
                i++
            }

            this.weight = new Tensor(wData, wShape)
            this.dW = new Tensor(alloc(wSize), wShape)

            // Bias: [outChannels]
            let bData = alloc(outChannels)
            this.bias = new Tensor(bData, [outChannels])
            this.dB = new Tensor(alloc(outChannels), [outChannels])
        }

        // ---------------------------------------------------------
        // Forward (fractionally strided convolution)
        // ---------------------------------------------------------
        forward(x: Tensor): Tensor {
            this.lastInput = x

            let batch = x.shape[0]
            let inC = x.shape[1]
            let dims = this.kernelShape.length

            // Compute output spatial dims
            let outSpatial = alloc(dims)
            let i = 0
            while (i < dims) {
                let inSize = x.shape[2 + i]
                let k = this.kernelShape[i]
                let s = this.stride[i]
                let p = this.padding[i]

                // Transposed conv output formula:
                outSpatial[i] = (inSize - 1) * s - 2 * p + k
                i++
            }

            // Output shape: [batch, outChannels, ...outSpatial]
            let outShape = alloc(2 + dims)
            outShape[0] = batch
            outShape[1] = this.outChannels
            i = 0
            while (i < dims) {
                outShape[2 + i] = outSpatial[i]
                i++
            }

            // Allocate output
            let total = 1
            i = 0
            while (i < outShape.length) {
                total *= outShape[i]
                i++
            }

            let outData = alloc(total)
            let out = new Tensor(outData, outShape)

            // Temp index arrays
            let idxX = alloc(2 + dims)
            let idxW = alloc(2 + dims)
            let idxOut = alloc(2 + dims)
            let kpos = alloc(dims)

            // Loop over input positions
            let p = 0
            let sizeX = x.data.length

            while (p < sizeX) {
                idxX = Tensor.unravelIndex(p, x.shape)

                let b = idxX[0]
                let ic = idxX[1]

                // Compute base output position
                let baseOut = alloc(dims)
                let d = 0
                while (d < dims) {
                    baseOut[d] = idxX[2 + d] * this.stride[d] - this.padding[d]
                    d++
                }

                // ND kernel loop
                let level = 0
                let reset = 0
                while (reset < dims) {
                    kpos[reset] = 0
                    reset++
                }

                while (level >= 0) {
                    if (level == dims) {
                        // Compute output index
                        let inside = true
                        let d2 = 0
                        while (d2 < dims) {
                            let oPos = baseOut[d2] + kpos[d2]
                            idxOut[2 + d2] = oPos
                            if (oPos < 0 || oPos >= outSpatial[d2]) inside = false
                            d2++
                        }

                        idxOut[0] = b

                        if (inside) {
                            let j = 0
                            while (j < this.outChannels) {
                                idxOut[1] = j

                                // Weight index
                                idxW[0] = ic
                                idxW[1] = j
                                let d3 = 0
                                while (d3 < dims) {
                                    idxW[2 + d3] = kpos[d3]
                                    d3++
                                }

                                let wVal = this.weight.get(idxW)
                                let outIdx = out.index(idxOut)
                                outData[outIdx] += x.data[p] * wVal
                                j++
                            }
                        }

                        level--
                        continue
                    }

                    if (kpos[level] < this.kernelShape[level]) {
                        kpos[level]++
                        level++
                    } else {
                        kpos[level] = 0
                        level--
                    }
                }

                p++
            }

            // Add bias
            let q = 0
            while (q < outData.length) {
                let idx = Tensor.unravelIndex(q, outShape)
                let oc = idx[1]
                outData[q] += this.bias.data[oc]
                q++
            }

            return out
        }

        // ---------------------------------------------------------
        // Backward
        // ---------------------------------------------------------
        backward(gradOut: Tensor): Tensor {
            let x = this.lastInput
            let dims = this.kernelShape.length

            // Zero gradients
            let i = 0
            while (i < this.dW.data.length) {
                this.dW.data[i] = 0
                i++
            }
            i = 0
            while (i < this.dB.data.length) {
                this.dB.data[i] = 0
                i++
            }

            // dB: sum over gradOut
            let q = 0
            while (q < gradOut.data.length) {
                let idx = Tensor.unravelIndex(q, gradOut.shape)
                let oc = idx[1]
                this.dB.data[oc] += gradOut.data[q]
                q++
            }

            // dX
            let dXdata = alloc(x.data.length)
            let dX = new Tensor(dXdata, x.shape.slice(0))

            // Temp index arrays
            let idxX = alloc(2 + dims)
            let idxW = alloc(2 + dims)
            let idxOut = alloc(2 + dims)
            let kpos = alloc(dims)

            // Loop over input positions for dW and dX
            let p = 0
            let sizeX = x.data.length

            while (p < sizeX) {
                idxX = Tensor.unravelIndex(p, x.shape)

                let b = idxX[0]
                let ic = idxX[1]

                // Base output position
                let baseOut = alloc(dims)
                let d = 0
                while (d < dims) {
                    baseOut[d] = idxX[2 + d] * this.stride[d] - this.padding[d]
                    d++
                }

                // ND kernel loop
                let level = 0
                let reset = 0
                while (reset < dims) {
                    kpos[reset] = 0
                    reset++
                }

                while (level >= 0) {
                    if (level == dims) {
                        let inside = true
                        let d2 = 0
                        while (d2 < dims) {
                            let oPos = baseOut[d2] + kpos[d2]
                            idxOut[2 + d2] = oPos
                            if (oPos < 0 || oPos >= gradOut.shape[2 + d2]) inside = false
                            d2++
                        }

                        idxOut[0] = b

                        if (inside) {
                            let j = 0
                            while (j < this.outChannels) {
                                idxOut[1] = j

                                let go = gradOut.get(idxOut)

                                // dW
                                idxW[0] = ic
                                idxW[1] = j
                                let d3 = 0
                                while (d3 < dims) {
                                    idxW[2 + d3] = kpos[d3]
                                    d3++
                                }
                                let wIdx = this.weight.index(idxW)
                                this.dW.data[wIdx] += x.data[p] * go

                                // dX
                                dXdata[p] += this.weight.data[wIdx] * go

                                j++
                            }
                        }

                        level--
                        continue
                    }

                    if (kpos[level] < this.kernelShape[level]) {
                        kpos[level]++
                        level++
                    } else {
                        kpos[level] = 0
                        level--
                    }
                }

                p++
            }

            return dX
        }
    }
}