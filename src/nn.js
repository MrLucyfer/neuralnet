const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

class NeuralNet {
    constructor() {
        this.netStruct = new Array();
        this.lr = 0.1;
    }

    add(layer) {
        this.netStruct.push(layer);
    }

    compile() {
      for(let i = 0; i < this.netStruct.length; i++) {
        this.netStruct[i].init();
        //console.log(this.netStruct[i].shape)
      }
      this.netStruct[this.netStruct.length - 1].output = true;
    }

    feedForward(X) {
        let output = this.netStruct[0].weightSum(X);
        for(let i = 1; i < this.netStruct.length; i++) {
            output = this.netStruct[i].weightSum(output);
        }
        return this.netStruct[this.netStruct.length - 1].sumOutput
    }

    async train(dataset, epochs=0) {
        const opt = tf.train.adam(this.lr);
        for(let i = 0; i < epochs; i++) {
            dataset = dataset.shuffle(3)
            await dataset.forEachAsync((data) => {
                const ys = tf.tensor1d(data.y);
                opt.minimize(() => {
                    const output = this.feedForward(data.X)
                    const error = tf.metrics.categoricalCrossentropy(ys, output);
                    return error.asScalar()
                })
            });
            console.log('Epoch', i);
        }

        console.log("Model trained!");
    }

    async saveModel(filename) {
        
        //Creamos schema del modelo
        let obj = {
            lr: this.lr,
            layers: []
        }

        this.netStruct.forEach((x, i) => {
            console.log(x)
            let data = {
                index: i,
                input: 0,
                shape: 0,
                activation: 0,
            }
            data.input = x.input;
            data.shape = x.shape;
            data.activation = x.activation;
            obj.layers.push(data)
        });

        //Guardamos los tensores en una archivo
        const writer = createWriter('weights.nba');
        for(let i = 0; i < this.netStruct.length; i++) {
            let data = await this.netStruct[i].weights.data();
            writer.print(data);
            writer.print('\n');
            data = await this.netStruct[i].bias.data();
            writer.print(data);
            writer.print('\n');
        }
        writer.close();
        writer.clear();
        saveJSON(obj, `${filename}.json`)
    }

    printLayers() {
        for(let layer of this.netStruct) {
            layer.print()
        }
    }

    predict(X) {
        const prediction = this.feedForward(X);
        prediction.print();
    } 
}

module.exports = {
  NeuralNet
}
