namespace TorchNew.Activations {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0, size)

        return arr
    }

    // Base interface (not required, but helpful)
    export interface Activation {
        forward(x: Tensor): Tensor
        backward(x: Tensor, gradOut: Tensor): Tensor
    }

    // ---------------------------------------------------------
    // ReLU
    // ---------------------------------------------------------
    export class ReLU implements Activation {

        forward(x: Tensor): Tensor {
            let outData = alloc(x.data.length)
            let i = 0
            while (i < x.data.length) {
                let v = x.data[i]
                outData[i] = v > 0 ? v : 0
                i++
            }
            return new Tensor(outData, x.shape.slice(0))
        }

        backward(x: Tensor, gradOut: Tensor): Tensor {
            let grad = alloc(x.data.length)
            let i = 0
            while (i < x.data.length) {
                grad[i] = x.data[i] > 0 ? gradOut.data[i] : 0
                i++
            }
            return new Tensor(grad, x.shape.slice(0))
        }
    }

    // ---------------------------------------------------------
    // Sigmoid
    // ---------------------------------------------------------
    export class Sigmoid implements Activation {

        forward(x: Tensor): Tensor {
            let outData = alloc(x.data.length)
            let i = 0
            while (i < x.data.length) {
                let v = x.data[i]
                outData[i] = 1 / (1 + Math.exp(-v))
                i++
            }
            return new Tensor(outData, x.shape.slice(0))
        }

        backward(x: Tensor, gradOut: Tensor): Tensor {
            let grad = alloc(x.data.length)
            let i = 0
            while (i < x.data.length) {
                let s = 1 / (1 + Math.exp(-x.data[i]))
                grad[i] = gradOut.data[i] * s * (1 - s)
                i++
            }
            return new Tensor(grad, x.shape.slice(0))
        }
    }

    function fastTanh(x: number): number {
        let e2x = Math.exp(2 * x)
        return (e2x - 1) / (e2x + 1)
    }

    // ---------------------------------------------------------
    // Tanh
    // ---------------------------------------------------------
    export class Tanh implements Activation {

        forward(x: Tensor): Tensor {
            let outData = alloc(x.data.length)
            let i = 0
            while (i < x.data.length) {
                outData[i] = fastTanh(x.data[i])
                i++
            }
            return new Tensor(outData, x.shape.slice(0))
        }

        backward(x: Tensor, gradOut: Tensor): Tensor {
            let grad = alloc(x.data.length)
            let i = 0
            while (i < x.data.length) {
                let t = fastTanh(x.data[i])
                grad[i] = gradOut.data[i] * (1 - t * t)
                i++
            }
            return new Tensor(grad, x.shape.slice(0))
        }
    }

    // ---------------------------------------------------------
    // GELU (approximate version)
    // ---------------------------------------------------------
    export class GELU implements Activation {

        forward(x: Tensor): Tensor {
            let outData = alloc(x.data.length)
            let i = 0
            while (i < x.data.length) {
                let v = x.data[i]
                let c = 0.044715
                let k = Math.sqrt(2 / Math.PI)
                let inner = k * (v + c * v * v * v)
                outData[i] = 0.5 * v * (1 + fastTanh(inner))
                i++
            }
            return new Tensor(outData, x.shape.slice(0))
        }

        backward(x: Tensor, gradOut: Tensor): Tensor {
            let grad = alloc(x.data.length)
            let i = 0
            while (i < x.data.length) {
                let v = x.data[i]
                let c = 0.044715
                let k = Math.sqrt(2 / Math.PI)
                let inner = k * (v + c * v * v * v)
                let t = fastTanh(inner)
                let sech2 = 1 - t * t
                let innerDeriv = k * (1 + 3 * c * v * v)
                let geluDeriv = 0.5 * (1 + t) + 0.5 * v * sech2 * innerDeriv

                grad[i] = gradOut.data[i] * geluDeriv
                i++
            }
            return new Tensor(grad, x.shape.slice(0))
        }
    }

    // ---------------------------------------------------------
    // LeakyReLU
    // ---------------------------------------------------------
    export class LeakyReLU implements Activation {
        alpha: number

        constructor(alpha: number) {
            this.alpha = alpha
        }

        forward(x: Tensor): Tensor {
            let outData = alloc(x.data.length)
            let i = 0
            while (i < x.data.length) {
                let v = x.data[i]
                outData[i] = v > 0 ? v : this.alpha * v
                i++
            }
            return new Tensor(outData, x.shape.slice(0))
        }

        backward(x: Tensor, gradOut: Tensor): Tensor {
            let grad = alloc(x.data.length)
            let i = 0
            while (i < x.data.length) {
                grad[i] = x.data[i] > 0 ? gradOut.data[i] : this.alpha * gradOut.data[i]
                i++
            }
            return new Tensor(grad, x.shape.slice(0))
        }
    }

    // ---------------------------------------------------------
    // Softplus
    // ---------------------------------------------------------
    export class Softplus implements Activation {

        forward(x: Tensor): Tensor {
            let outData = alloc(x.data.length)
            let i = 0
            while (i < x.data.length) {
                let v = x.data[i]
                outData[i] = Math.log(1 + Math.exp(v))
                i++
            }
            return new Tensor(outData, x.shape.slice(0))
        }

        backward(x: Tensor, gradOut: Tensor): Tensor {
            let grad = alloc(x.data.length)
            let i = 0
            while (i < x.data.length) {
                let v = x.data[i]
                let s = 1 / (1 + Math.exp(-v)) // sigmoid
                grad[i] = gradOut.data[i] * s
                i++
            }
            return new Tensor(grad, x.shape.slice(0))
        }
    }
}