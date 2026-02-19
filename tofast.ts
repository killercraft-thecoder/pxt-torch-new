namespace TorchNew {

    export function ToFast(t: Tensor): FastTensor {
        // Reject 0D tensors
        if (t.shape.length === 0) {
            console.log("ToFast ERROR: Cannot convert 0D tensor")
            return null
        }

        // If 1D: [N] → [1][N]
        if (t.shape.length === 1) {
            let row: number[] = []
            for (let i = 0; i < t.data.length; i++) {
                row.push(t.data[i])
            }
            return new FastTensor([row])
        }

        // If 2D: [A, B] → [A][B]
        if (t.shape.length === 2) {
            let rows = t.shape[0]
            let cols = t.shape[1]
            let out: number[][] = []
            let idx = 0

            for (let r = 0; r < rows; r++) {
                let row: number[] = []
                for (let c = 0; c < cols; c++) {
                    row.push(t.data[idx])
                    idx++
                }
                out.push(row)
            }

            return new FastTensor(out)
        }

        // ND case: flatten all dims except last
        // shape [D1, D2, ..., Dk, F] → [D1*D2*...*Dk][F]
        let lastDim = t.shape[t.shape.length - 1]
        let batch = 1

        for (let i = 0; i < t.shape.length - 1; i++) {
            batch *= t.shape[i]
        }

        let out: number[][] = []
        let idx = 0

        for (let r = 0; r < batch; r++) {
            let row: number[] = []
            for (let c = 0; c < lastDim; c++) {
                row.push(t.data[idx])
                idx++
            }
            out.push(row)
        }

        return new FastTensor(out)
    }
}