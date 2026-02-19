namespace TorchNew {

    function alloc(size: number): number[] {
        let arr = Array.repeat(0, size)
        return arr
    }
    
    export class CrossEntropyLoss {
        lastInput: Tensor
        lastTarget: Tensor
        lastSoftmax: Tensor

        constructor() { }

        // ---------------------------------------------------------
        // Forward: logits + class indices -> scalar loss
        // ---------------------------------------------------------
        forward(logits: Tensor, target: Tensor): Tensor {
            // logits shape: [batch, classes]
            // target shape: [batch] (integer class indices)

            this.lastInput = logits
            this.lastTarget = target

            let batch = logits.shape[0]
            let classes = logits.shape[1]

            let lossSum = 0

            let i = 0
            while (i < batch) {
                // Compute stable max
                let rowStart = i * classes
                let maxVal = logits.data[rowStart]
                let j = 1
                while (j < classes) {
                    let v = logits.data[rowStart + j]
                    if (v > maxVal) maxVal = v
                    j++
                }

                // Compute log-sum-exp
                let sumExp = 0
                j = 0
                while (j < classes) {
                    sumExp += Math.exp(logits.data[rowStart + j] - maxVal)
                    j++
                }

                let logSumExp = Math.log(sumExp) + maxVal

                // Pick correct class logit
                let y = target.data[i]
                let correctLogit = logits.data[rowStart + y]

                // Loss = logSumExp - correctLogit
                lossSum += logSumExp - correctLogit

                i++
            }

            let loss = lossSum / batch

            // Return scalar tensor
            return new Tensor([loss], [1])
        }

        // ---------------------------------------------------------
        // Backward: dL/dlogits = softmax(logits) - one_hot(target)
        // ---------------------------------------------------------
        backward(): Tensor {
            let logits = this.lastInput
            let target = this.lastTarget

            let batch = logits.shape[0]
            let classes = logits.shape[1]

            let gradData = alloc(logits.data.length)

            // Compute softmax for backward
            let soft = this.softmax(logits)

            let i = 0
            while (i < batch) {
                let rowStart = i * classes
                let y = target.data[i]

                let j = 0
                while (j < classes) {
                    let s = soft.data[rowStart + j]
                    let indicator = (j == y) ? 1 : 0
                    gradData[rowStart + j] = (s - indicator) / batch
                    j++
                }

                i++
            }

            return new Tensor(gradData, logits.shape.slice(0))
        }

        // ---------------------------------------------------------
        // Internal stable softmax (same as Softmax layer)
        // ---------------------------------------------------------
        softmax(x: Tensor): Tensor {
            let batch = x.shape[0]
            let classes = x.shape[1]

            let outData = alloc(x.data.length)

            let i = 0
            while (i < batch) {
                let rowStart = i * classes

                // max
                let maxVal = x.data[rowStart]
                let j = 1
                while (j < classes) {
                    let v = x.data[rowStart + j]
                    if (v > maxVal) maxVal = v
                    j++
                }

                // exp
                let sumExp = 0
                j = 0
                while (j < classes) {
                    let e = Math.exp(x.data[rowStart + j] - maxVal)
                    outData[rowStart + j] = e
                    sumExp += e
                    j++
                }

                // normalize
                j = 0
                while (j < classes) {
                    outData[rowStart + j] /= sumExp
                    j++
                }

                i++
            }

            return new Tensor(outData, x.shape.slice(0))
        }
    }
}