namespace TorchNew {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0, size)
        return arr
    }



    export class ConvND {
        inChannels: number
        outChannels: number
        kernelShape: number[]
        padding: number[]
        stride: number[]
        weight: Tensor
        bias: Tensor

        constructor(inChannels: number, outChannels: number,
            kernelShape: number[], padding: number[], stride: number[]) {

            this.inChannels = inChannels
            this.outChannels = outChannels
            this.kernelShape = kernelShape.slice(0)
            this.padding = padding.slice(0)
            this.stride = stride.slice(0)

            let dims = kernelShape.length

            // Weight shape: [outChannels, inChannels, ...kernelShape]
            let wsize = outChannels * inChannels
            let i = 0
            while (i < dims) {
                wsize *= kernelShape[i]
                i++
            }

            let wdata = alloc(wsize)

            // Xavier-like init
            let scale = 1 / Math.sqrt(inChannels)
            i = 0
            while (i < wsize) {
                wdata[i] = (Math.random() - 0.5) * 2 * scale
                i++
            }

            let wshape = alloc(2 + dims)
            wshape[0] = outChannels
            wshape[1] = inChannels
            i = 0
            while (i < dims) {
                wshape[2 + i] = kernelShape[i]
                i++
            }

            this.weight = new Tensor(wdata, wshape)

            // Bias shape: [outChannels]
            let bdata = alloc(outChannels)
            this.bias = new Tensor(bdata, [outChannels])
        }

        forward(x: Tensor): Tensor {
            // x shape: [batch, inChannels, ...spatial]
            let batch = x.shape[0]
            let cIn = x.shape[1]
            let dims = this.kernelShape.length

            // Compute output spatial dims
            let outSpatial = alloc(dims)
            let i = 0
            while (i < dims) {
                let inSize = x.shape[2 + i]
                let k = this.kernelShape[i]
                let pad = this.padding[i]
                let s = this.stride[i]

                outSpatial[i] = Math.idiv(inSize + 2 * pad - k, s) + 1
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

            // Temporary index arrays
            let idxOut = alloc(2 + dims)
            let idxX = alloc(2 + dims)
            let idxW = alloc(2 + dims)

            // Convolution loops
            let b = 0
            while (b < batch) {
                idxOut[0] = b
                idxX[0] = b

                let oc = 0
                while (oc < this.outChannels) {
                    idxOut[1] = oc
                    idxW[0] = oc

                    // Iterate over output spatial dims
                    this._loopOutputDims(
                        0, dims,
                        idxOut, outSpatial,
                        x, out,
                        idxX, idxW,
                        oc
                    )

                    oc++
                }
                b++
            }

            return out
        }

        // Iterates over output spatial dims (non-recursive wrapper)
        _loopOutputDims(dim: number, dims: number,
            idxOut: number[], outSpatial: number[],
            x: Tensor, out: Tensor,
            idxX: number[], idxW: number[],
            oc: number): void {

            // We simulate recursion manually using a stack
            let stackDim = alloc(dims)
            let stackPos = alloc(dims)

            let d = 0
            while (d < dims) {
                stackDim[d] = 0
                stackPos[d] = 0
                d++
            }

            let level = 0

            while (level >= 0) {
                if (level == dims) {
                    // Compute convolution at this output position
                    let sum = this.bias.data[oc]

                    let ic = 0
                    while (ic < this.inChannels) {
                        idxX[1] = ic
                        idxW[1] = ic

                        sum = this._applyKernel(
                            dims, idxOut, idxX, idxW,
                            x, sum
                        )

                        ic++
                    }

                    out.set(idxOut, sum)

                    level--
                    continue
                }

                if (stackPos[level] < outSpatial[level]) {
                    idxOut[2 + level] = stackPos[level]
                    stackPos[level]++
                    level++
                } else {
                    stackPos[level] = 0
                    level--
                }
            }
        }

        // Applies kernel at a single output position
        _applyKernel(dims: number,
            idxOut: number[], idxX: number[], idxW: number[],
            x: Tensor, sum: number): number {

            // Manual ND kernel loops
            let kpos = alloc(dims)
            let level = 0

            while (level >= 0) {
                if (level == dims) {
                    // Compute input index
                    let inside = true
                    let d = 0
                    while (d < dims) {
                        let outPos = idxOut[2 + d]
                        let k = kpos[d]
                        let pad = this.padding[d]
                        let s = this.stride[d]

                        let inPos = outPos * s + k - pad
                        idxX[2 + d] = inPos

                        if (inPos < 0 || inPos >= x.shape[2 + d]) {
                            inside = false
                        }

                        idxW[2 + d] = k
                        d++
                    }

                    if (inside) {
                        let xVal = x.get(idxX)
                        let wVal = this.weight.get(idxW)
                        sum += xVal * wVal
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

            return sum
        }

        backward(dY: Tensor, x: Tensor): { dX: Tensor, dW: Tensor, dB: Tensor } {
            // Shapes:
            // x:  [batch, inChannels, ...inSpatial]
            // dY: [batch, outChannels, ...outSpatial]
            // W:  [outChannels, inChannels, ...kernelShape]

            let batch = x.shape[0]
            let cIn = this.inChannels
            let cOut = this.outChannels
            let dims = this.kernelShape.length

            // Allocate dX (same shape as x)
            let sizeDX = 1
            let i = 0
            while (i < x.shape.length) {
                sizeDX *= x.shape[i]
                i++
            }
            let dXdata = alloc(sizeDX)
            let dX = new Tensor(dXdata, x.shape.slice(0))

            // Allocate dW (same shape as weight)
            let sizeDW = 1
            i = 0
            while (i < this.weight.shape.length) {
                sizeDW *= this.weight.shape[i]
                i++
            }
            let dWdata = alloc(sizeDW)
            let dW = new Tensor(dWdata, this.weight.shape.slice(0))

            // Allocate dB (same shape as bias)
            let dBdata = alloc(cOut)
            let dB = new Tensor(dBdata, [cOut])

            // Temp index arrays
            let idxY = alloc(2 + dims)   // [b, oc, o...]
            let idxX = alloc(2 + dims)   // [b, ic, i...]
            let idxW = alloc(2 + dims)   // [oc, ic, k...]
            let kpos = alloc(dims)       // kernel position per dim

            // Total elements in dY
            let sizeDY = dY.data.length
            let p = 0

            while (p < sizeDY) {
                // Unravel flat index p into dY indices
                idxY = Tensor.unravelIndex(p, dY.shape)
                let b = idxY[0]
                let oc = idxY[1]

                let grad = dY.data[p]

                // 1) dB[oc] += dY[b, oc, ...]
                dB.data[oc] += grad

                // 2) Loop over input channels and kernel positions
                let ic = 0
                while (ic < cIn) {
                    idxX[0] = b
                    idxX[1] = ic
                    idxW[0] = oc
                    idxW[1] = ic

                    // ND kernel loop using manual stack
                    let level = 0
                    let kmax = this.kernelShape
                    let kcur = kpos
                    let reset = 0
                    while (reset < dims) {
                        kcur[reset] = 0
                        reset++
                    }

                    while (level >= 0) {
                        if (level == dims) {
                            // Compute input spatial index for this kernel position
                            let inside = true
                            let d = 0
                            while (d < dims) {
                                let oPos = idxY[2 + d]
                                let k = kcur[d]
                                let pad = this.padding[d]
                                let s = this.stride[d]

                                let iPos = oPos * s + k - pad
                                idxX[2 + d] = iPos
                                idxW[2 + d] = k

                                if (iPos < 0 || iPos >= x.shape[2 + d]) {
                                    inside = false
                                }
                                d++
                            }

                            if (inside) {
                                // dW[oc, ic, k...] += dY * x[b, ic, i...]
                                let xVal = x.get(idxX)
                                let wGradIndex = dW.index(idxW)
                                dW.data[wGradIndex] += grad * xVal

                                // dX[b, ic, i...] += dY * W[oc, ic, k...]
                                let wVal = this.weight.get(idxW)
                                let xGradIndex = dX.index(idxX)
                                dX.data[xGradIndex] += grad * wVal
                            }

                            level--
                            continue
                        }

                        if (kcur[level] < kmax[level]) {
                            kcur[level]++
                            level++
                        } else {
                            kcur[level] = 0
                            level--
                        }
                    }

                    ic++
                }

                p++
            }

            return {
                dX: dX,
                dW: dW,
                dB: dB
            }
        }
    }



}