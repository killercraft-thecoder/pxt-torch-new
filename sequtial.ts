namespace TorchNew {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0, size)
        return arr
    }

    export class Sequential {
        layers: any[]   // Each layer must have forward(), backward(), and optionally parameters()

        constructor(layers: any[]) {
            this.layers = layers
        }

        // Forward pass through all layers
        forward(x: Tensor): Tensor {
            let out = x
            let i = 0
            while (i < this.layers.length) {
                out = this.layers[i].forward(out)
                i++
            }
            return out
        }

        // Backward pass through all layers (reverse order)
        backward(grad: Tensor): Tensor {
            let g = grad
            let i = this.layers.length - 1
            while (i >= 0) {
                // Some layers need the original input for backward
                // So we store last input inside each layer during forward
                if (this.layers[i].lastInput) {
                    g = this.layers[i].backward(this.layers[i].lastInput, g)
                } else {
                    // Layers like Linear or ConvND already store their own input
                    g = this.layers[i].backward(g)
                }
                i--
            }
            return g
        }

        // Collect all parameters from all layers
        parameters(): Tensor[] {
            let params: Tensor[] = []
            let i = 0
            while (i < this.layers.length) {
                let layer = this.layers[i]
                if (layer.weight) params.push(layer.weight)
                if (layer.bias) params.push(layer.bias)
                i++
            }
            return params
        }

        // Collect all gradients from all layers
        gradients(): Tensor[] {
            let grads: Tensor[] = []
            let i = 0
            while (i < this.layers.length) {
                let layer = this.layers[i]
                if (layer.dW) grads.push(layer.dW)
                if (layer.dB) grads.push(layer.dB)
                i++
            }
            return grads
        }
    }
}