namespace TorchNew {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0,size)
        return arr
    }

    export class Tensor {
        data: number[]
        shape: number[]
        strides: number[]

        constructor(data: number[], shape: number[]) {
            this.data = data
            this.shape = shape
            this.strides = Tensor.computeStrides(shape)
        }

        // Compute strides for ND indexing
        static computeStrides(shape: number[]): number[] {
            let n = shape.length
            let strides = alloc(n)
            let acc = 1
            let i = n - 1

            while (i >= 0) {
                strides[i] = acc
                acc *= shape[i]
                i--
            }

            return strides
        }

        // Convert ND index → flat index
        index(indices: number[]): number {
            let offset = 0
            let i = 0

            while (i < indices.length) {
                offset += indices[i] * this.strides[i]
                i++
            }

            return offset
        }

        get(indices: number[]): number {
            return this.data[this.index(indices)]
        }

        set(indices: number[], value: number): void {
            this.data[this.index(indices)] = value
        }

        reshape(newShape: number[]): Tensor {
            let newSize = 1
            let i = 0

            while (i < newShape.length) {
                newSize *= newShape[i]
                i++
            }

            if (newSize != this.data.length) {
                return null
            }

            return new Tensor(this.data.slice(0), newShape.slice(0))
        }

        static unravelIndex(flat: number, shape: number[]): number[] {
            let rank = shape.length
            let out = alloc(rank)
            let i = rank - 1

            while (i >= 0) {
                let dim = shape[i]
                out[i] = flat % dim
                flat = Math.idiv(flat, dim)
                i--
            }

            return out
        }

        transpose(a: number, b: number): Tensor {
            let newShape = this.shape.slice(0)
            let t = newShape[a]
            newShape[a] = newShape[b]
            newShape[b] = t

            let out = alloc(this.data.length)
            let outTensor = new Tensor(out, newShape)

            let total = this.data.length
            let flat = 0

            while (flat < total) {
                let idx = Tensor.unravelIndex(flat, this.shape)
                let tmp = idx[a]
                idx[a] = idx[b]
                idx[b] = tmp

                outTensor.set(idx, this.data[flat])
                flat++
            }

            return outTensor
        }

        add(other: Tensor): Tensor {
            let size = this.data.length
            let out = alloc(size)
            let i = 0

            while (i < size) {
                out[i] = this.data[i] + other.data[i]
                i++
            }

            return new Tensor(out, this.shape.slice(0))
        }

        sub(other: Tensor): Tensor {
            let size = this.data.length
            let out = alloc(size)
            let i = 0

            while (i < size) {
                out[i] = this.data[i] - other.data[i]
                i++
            }

            return new Tensor(out, this.shape.slice(0))
        }

        extract2D(prefix: number[], rows: number, cols: number): number[] {
            let out = alloc(rows * cols)
            let r = 0

            while (r < rows) {
                let c = 0
                while (c < cols) {
                    let idx = prefix.slice(0)
                    idx.push(r)
                    idx.push(c)
                    out[r * cols + c] = this.get(idx)
                    c++
                }
                r++
            }

            return out
        }

        write2D(prefix: number[], rows: number, cols: number, src: number[]): void {
            let r = 0

            while (r < rows) {
                let c = 0
                while (c < cols) {
                    let idx = prefix.slice(0)
                    idx.push(r)
                    idx.push(c)
                    this.set(idx, src[r * cols + c])
                    c++
                }
                r++
            }
        }

        static matmul2D(a: number[], aRows: number, aCols: number,
            b: number[], bRows: number, bCols: number): number[] {

            let out = alloc(aRows * bCols)
            let r = 0

            while (r < aRows) {
                let c = 0
                while (c < bCols) {
                    let sum = 0
                    let k = 0

                    while (k < aCols) {
                        sum += a[r * aCols + k] * b[k * bCols + c]
                        k++
                    }

                    out[r * bCols + c] = sum
                    c++
                }
                r++
            }

            return out
        }



        matmul(other: Tensor): Tensor {
            let aShape = this.shape
            let bShape = other.shape

            let aRank = aShape.length
            let bRank = bShape.length

            // Last two dims must be matrix dims
            let aRows = aShape[aRank - 2]
            let aCols = aShape[aRank - 1]
            let bRows = bShape[bRank - 2]
            let bCols = bShape[bRank - 1]

            if (aCols != bRows) {
                return null
            }

            // Determine batch shape (everything except last 2 dims)
            let batchRank = aRank - 2
            let batchShape = Array.repeat(0,batchRank)
            let i = 0

            while (i < batchRank) {
                batchShape[i] = aShape[i]
                i++
            }

            // Output shape = batch + [aRows, bCols]
            let outShape = Array.repeat(0,batchRank + 2)
            i = 0

            while (i < batchRank) {
                outShape[i] = batchShape[i]
                i++
            }

            outShape[batchRank] = aRows
            outShape[batchRank + 1] = bCols

            // Allocate output tensor
            let total = 1
            i = 0
            while (i < outShape.length) {
                total *= outShape[i]
                i++
            }

            let outData = alloc(total)
            let out = new Tensor(outData, outShape)

            // Compute number of batches
            let batchSize = 1
            i = 0
            while (i < batchRank) {
                batchSize *= batchShape[i]
                i++
            }

            // Loop over all batches
            let bIndex = 0

            while (bIndex < batchSize) {
                // Convert flat batch index → ND prefix
                let prefix = Tensor.unravelIndex(bIndex, batchShape)

                // Extract 2D slices
                let a2 = this.extract2D(prefix, aRows, aCols)
                let b2 = other.extract2D(prefix, bRows, bCols)

                // Multiply 2D slices
                let result2 = Tensor.matmul2D(a2, aRows, aCols, b2, bRows, bCols)

                // Write result back into output tensor
                out.write2D(prefix, aRows, bCols, result2)

                bIndex++
            }

            return out
        }
    }
}