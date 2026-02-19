namespace TorchNew {

    // Convert FastTensor (2D) → full Tensor (ND) with a given shape
    export function FromFast(ft: FastTensor, shape: number[]): Tensor {
        // Validate shape
        if (shape.length === 0) {
            console.log("FromFast ERROR: Cannot create 0D tensor")
            return null
        }

        // Flatten FastTensor.data (number[][]) into number[]
        let flat: number[] = []
        for (let r = 0; r < ft.data.length; r++) {
            for (let c = 0; c < ft.data[0].length; c++) {
                flat.push(ft.data[r][c])
            }
        }

        // Validate total size
        let expected = 1
        for (let i = 0; i < shape.length; i++) {
            expected *= shape[i]
        }

        if (expected !== flat.length) {
            console.log("FromFast ERROR: Shape mismatch. Expected size " + expected + " but got " + flat.length)
            return null
        }

        // Create full Tensor
        return new Tensor(flat, shape.slice(0))
    }
}