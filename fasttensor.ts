namespace TorchNew {
    export interface FastShape {
        rows: number;
        columns: number;
    }

    type Function = (x: number) => number

    /**
        * Represents a multi-dimensional tensor for number[][] computations.
        */
    export class FastTensor {
        shape: FastShape

        /** 
         * The Data Of the Tensor
        */
        data: number[][];
        /**
        * Creates a new tensor from a 2D `Matrix`.
        * @param data A 2D `Matrix` representing the tensor values.
        */
        constructor(data: number[][]) {
            this.data = data;
            this.shape = { rows: data.length, columns: data[0].length }
        }

        /**
         * Returns a new tensor with each element rounded to the nearest integer.
         *
         * @returns {FastTensor} A tensor with rounded values.
         */
        round(): FastTensor {
            return this.applyFunction((x:number) => Math.round(x))
        }

        /**
         * Returns a new tensor with each element rounded down to the nearest whole number.
         *
         * @returns {FastTensor} A tensor with floored values.
         */
        floor(): FastTensor {
            return this.applyFunction((x:number) => Math.floor(x))
        }

        /**
         * Flattens the 2D tensor into a 1D array in row-major order.
         *
         * @returns {number[]} A flat array containing all elements of the tensor.
         */
        flat(): number[] {
            let data: number[] = []
            this.data.forEach((a) => a.forEach((b) => data.push(b)))
            return data
        }

        /**
        * Performs matrix multiplication (A * B) and returns the resulting tensor.
        * @param other The tensor to multiply with.
        * @returns The resulting tensor, or `null` if dimensions do not match.
        */
        matmul(other: FastTensor): FastTensor | null {
            let temp1 = this.data; // Ensure a true copy
            let temp2 = other.data; // Prevent referencing original tensor
            let rowsA = temp1.length;
            let colsA = temp1[0].length;
            let rowsB = temp2.length;
            let colsB = temp2[0].length;

            if (colsA !== rowsB) {
                return null; // Dimension mismatch
            }

            let result: number[][] = [];

            // Optimized number[] multiplication
            for (let r = 0; r < rowsA; r++) { // Process row-wise first
                for (let i = 0; i < colsA; i++) {
                    let temp3 = temp1[r][i]; // Store lookup value for row
                    for (let c = 0; c < colsB; c++) {
                        result[r][c] += temp3 * temp2[i][c]; // Perform multiplication
                    }
                }
            }
            return new FastTensor(result);
        }
        /**
        * Applies a function to every element in the tensor and returns a new transformed tensor.
        * @param func The function to apply to each tensor element.
        * @returns A new tensor with transformed values.
        */
        applyFunction(func: Function): FastTensor {
            let data = this.data;
            let result = data.map(row => row.map(func)); // Direct transformation without extra storage
            return new TorchNew.FastTensor(result);
        }

        /**
        * Adds another tensor element-wise and returns the resulting tensor.
        * @param other The tensor to add.
        * @returns The resulting tensor after addition.
        */
        add(other: FastTensor): FastTensor {
            let rows = Math.min(this.data.length, other.data.length);
            let cols = Math.min(this.data[0].length, other.data[0].length);

            // Manual preallocation to prevent dynamic resizing overhead
            let result: number[][] = [];
            let data1 = this.data;
            let data2 = other.data;
            // Optimized addition loop
            for (let r = 0; r < rows; r++) {
                for (let c = 0; c < cols; c++) {
                    result[r][c] = data1[r][c] + data2[r][c]; // Direct assignment avoids push overhead
                }
            }

            return new TorchNew.FastTensor(result);
        }

        /**
        *Subtracts another tensor element-wise and returns the resulting tensor.
        * @param other The tensor to subtract.
        * @returns The resulting tensor after subtraction.
        */
        sub(other: FastTensor): FastTensor {
            let rows = Math.min(this.data.length, other.data.length);
            let cols = Math.min(this.data[0].length, other.data[0].length);

            // Manual array allocation without `new Array()`
            let result: number[][] = [];
            for (let r = 0; r < rows; r++) {
                result[r] = [];  // No `new Array()`, just an empty array
                for (let c = 0; c < cols; c++) {
                    result[r][c] = this.data[r][c] - other.data[r][c];
                }
            }

            return new TorchNew.FastTensor(result);
        }

        /**
        * Computes the sum of all elements in the tensor.
        * @returns The sum of all tensor elements.
        */
        sum(): number {
            return this.data.reduce((acc: number, row: number[]) => acc + row.reduce((rowAcc: number, value: number) => rowAcc + value, 0), 0);
        }
    }
}