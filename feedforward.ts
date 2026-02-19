namespace TorchNew {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0, size)
        return arr
    }
    
    export class FeedForward {
        layer1: any
        activation: any
        layer2: any

        lastInput: Tensor
        lastHidden: Tensor

        constructor(inputDim: number, hiddenDim: number, outputDim: number, activation: any) {
            this.layer1 = new Linear(inputDim, hiddenDim)
            this.activation = activation
            this.layer2 = new Linear(hiddenDim, outputDim)
        }

        // ---------------------------------------------------------
        // Forward
        // ---------------------------------------------------------
        forward(x: Tensor): Tensor {
            this.lastInput = x

            let h1 = this.layer1.forward(x)
            let h2 = this.activation.forward(h1)
            this.lastHidden = h2

            let out = this.layer2.forward(h2)
            return out
        }

        // ---------------------------------------------------------
        // Backward
        // ---------------------------------------------------------
        backward(gradOut: Tensor): Tensor {
            // Backprop through second linear
            let g2 = this.layer2.backward(gradOut)

            // Backprop through activation
            let gAct = this.activation.backward(g2)

            // Backprop through first linear
            let g1 = this.layer1.backward(gAct)

            return g1
        }
    }
}