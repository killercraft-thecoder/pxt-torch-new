namespace TorchNew {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0, size)
        return arr
    }

    export class Residual {
        layer: any
        lastInput: Tensor
        lastLayerOut: Tensor

        constructor(layer: any) {
            this.layer = layer
        }

        // ---------------------------------------------------------
        // Forward: y = x + layer(x)
        // ---------------------------------------------------------
        forward(x: Tensor): Tensor {
            this.lastInput = x

            let outLayer = this.layer.forward(x)
            this.lastLayerOut = outLayer

            let size = x.data.length
            let outData = alloc(size)

            let i = 0
            while (i < size) {
                outData[i] = x.data[i] + outLayer.data[i]
                i++
            }

            return new Tensor(outData, x.shape.slice(0))
        }

        // ---------------------------------------------------------
        // Backward: dX = gradOut + dLayer
        // ---------------------------------------------------------
        backward(gradOut: Tensor): Tensor {
            let gradLayer = this.layer.backward(gradOut)

            let size = gradOut.data.length
            let dXdata = alloc(size)

            let i = 0
            while (i < size) {
                dXdata[i] = gradOut.data[i] + gradLayer.data[i]
                i++
            }

            return new Tensor(dXdata, gradOut.shape.slice(0))
        }
    }
}