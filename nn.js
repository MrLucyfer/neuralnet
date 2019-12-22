function shuffle(array) {
    array.sort(() => Math.random() - 0.5);
  }

class NeuralNet {
    constructor() {
        this.netStruct = new Array();
        this.lr = 0.1;
    }

    add(layer) {
        this.netStruct.push(layer);
    }

    compile() {
        this.netStruct[0].init();
        for(let i = 1; i < this.netStruct.length; i++) {
            this.netStruct[i].init(this.netStruct[i-1].shape)
        }
    }
    feedForward(X) {
        let output = this.netStruct[0].weightSum(X);
        for(let i = 1; i < this.netStruct.length; i++) {
            output = this.netStruct[i].weightSum(output);
        }
        return this.netStruct[this.netStruct.length - 1].output
    }

    async train(dataset, epochs=0) {
        const opt = tf.train.adam(this.lr);
        for(let i = 0; i < epochs; i++) {
            dataset = dataset.shuffle(3)
            await dataset.forEachAsync(({X, y}) => {
                const ys = tf.tensor1d(y);
                opt.minimize(() => {
                    const output = this.feedForward(X)
                    const error = tf.metrics.categoricalCrossentropy(ys, output);
                    return error.asScalar()
                })
            });
            console.log('Epoch', i);
        }

        console.log("Model trained!");
    }

    predict(X) {
        this.feedForward(X).print()
    } 

    printLayers() {
        for(let layer of this.netStruct) {
            console.log("Shape of the layer:", layer.weights.shape)
            layer.print();
        }
    } 
}