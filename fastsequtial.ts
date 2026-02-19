namespace TorchNew {

    export class FastSequential {
        layers: any[]

        constructor(layers: any[]) {
            this.layers = layers
        }

        // Forward pass through all fast layers
        forward(x: FastTensor): FastTensor {
            let out = x
            for (let i = 0; i < this.layers.length; i++) {
                out = this.layers[i].forward(out)
            }
            return out
        }

        // Backward pass (reverse order)
        backward(grad: FastTensor, x: FastTensor): FastTensor {
            // For FastSequential, we assume each layer stores its own input
            // OR the user passes the correct x for the first layer.
            let g = grad

            // We need to track inputs for each layer
            // So we require each layer to store lastInput during forward
            for (let i = this.layers.length - 1; i >= 0; i--) {
                let layer = this.layers[i]

                if (!layer.lastInput) {
                    // If a layer didn't store its input, we cannot backprop
                    // Fast layers should ALWAYS store lastInput
                    console.log("FastSequential WARNING: layer missing lastInput")
                    g = layer.backward(g)
                } else {
                    g = layer.backward(g, layer.lastInput)
                }
            }

            return g
        }

        // Collect parameters from all fast layers
        parameters(): FastTensor[] {
            let params: FastTensor[] = []
            for (let i = 0; i < this.layers.length; i++) {
                let L = this.layers[i]
                if (L.weight) params.push(L.weight)
                if (L.bias) params.push(L.bias)
            }
            return params
        }

        // Collect gradients from all fast layers
        gradients(): FastTensor[] {
            let grads: FastTensor[] = []
            for (let i = 0; i < this.layers.length; i++) {
                let L = this.layers[i]
                if (L.dW) grads.push(L.dW)
                if (L.db) grads.push(L.db)
            }
            return grads
        }
    }
}