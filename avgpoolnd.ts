namespace TorchNew {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0, size)
        return arr
    }
    
    export class AvgPoolND {
        kernelShape: number[]
        stride: number[]
        padding: number[]
        lastInput:Tensor;

        constructor(kernelShape: number[], stride: number[], padding: number[]) {
            this.kernelShape = kernelShape.slice(0)
            this.stride = stride.slice(0)
            this.padding = padding.slice(0)
        }

        // ---------------------------------------------------------
        // Forward pass
        // ---------------------------------------------------------
        forward(x: Tensor): Tensor {
            // x shape: [batch, channels, ...spatial]
            let batch = x.shape[0]
            let channels = x.shape[1]
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

            // Output shape: [batch, channels, ...outSpatial]
            let outShape = alloc(2 + dims)
            outShape[0] = batch
            outShape[1] = channels
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
            let idxOut = alloc(2 + dims)
            let idxX = alloc(2 + dims)
            let kpos = alloc(dims)

            // Loop over all output positions
            let p = 0
            let sizeOut = out.data.length

            while (p < sizeOut) {
                idxOut = Tensor.unravelIndex(p, outShape)

                let b = idxOut[0]
                let c = idxOut[1]

                // Compute average over kernel window
                let sum = 0
                let count = 0

                // ND kernel loop using manual stack
                let level = 0
                let reset = 0
                while (reset < dims) {
                    kpos[reset] = 0
                    reset++
                }

                while (level >= 0) {
                    if (level == dims) {
                        // Compute input index
                        let inside = true
                        let d = 0
                        while (d < dims) {
                            let oPos = idxOut[2 + d]
                            let k = kpos[d]
                            let pad = this.padding[d]
                            let s = this.stride[d]

                            let iPos = oPos * s + k - pad
                            idxX[2 + d] = iPos

                            if (iPos < 0 || iPos >= x.shape[2 + d]) {
                                inside = false
                            }
                            d++
                        }

                        idxX[0] = b
                        idxX[1] = c

                        if (inside) {
                            sum += x.get(idxX)
                        }
                        count++

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

                out.data[p] = sum / count
                p++
            }

            this.lastInput = x
            return out
        }

        // ---------------------------------------------------------
        // Backward pass
        // ---------------------------------------------------------
        backward(gradOut: Tensor): Tensor {
            let x = this.lastInput
            let dims = this.kernelShape.length

            // Allocate dX
            let sizeDX = x.data.length
            let dXdata = alloc(sizeDX)
            let dX = new Tensor(dXdata, x.shape.slice(0))

            let idxOut = alloc(2 + dims)
            let idxX = alloc(2 + dims)
            let kpos = alloc(dims)

            let sizeOut = gradOut.data.length
            let p = 0

            while (p < sizeOut) {
                idxOut = Tensor.unravelIndex(p, gradOut.shape)

                let b = idxOut[0]
                let c = idxOut[1]
                let g = gradOut.data[p]

                // ND kernel loop
                let level = 0
                let reset = 0
                while (reset < dims) {
                    kpos[reset] = 0
                    reset++
                }

                let count = 1
                let i = 0
                while (i < dims) {
                    count *= this.kernelShape[i]
                    i++
                }

                while (level >= 0) {
                    if (level == dims) {
                        let inside = true
                        let d = 0
                        while (d < dims) {
                            let oPos = idxOut[2 + d]
                            let k = kpos[d]
                            let pad = this.padding[d]
                            let s = this.stride[d]

                            let iPos = oPos * s + k - pad
                            idxX[2 + d] = iPos

                            if (iPos < 0 || iPos >= x.shape[2 + d]) {
                                inside = false
                            }
                            d++
                        }

                        idxX[0] = b
                        idxX[1] = c

                        if (inside) {
                            let idx = dX.index(idxX)
                            dX.data[idx] += g / count
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