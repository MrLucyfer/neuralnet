const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');

class Layer {
    constructor(shape, activation = 'relu', input=false) {
        this.input = input;
        this.shape = shape;
        this.output = false;
        this.weights;
        this.activation = activation;
        this.bias = tf.variable(tf.randomUniform([this.shape[1]], -1, 1));
        this.sumOutput;
        this.variables;
    }

    init() {
        this.weights = tf.variable(tf.randomUniform(this.shape, -1, 1),true);
    }

    weightSum(input) {
        if(this.input) {
            input = tf.tensor1d(input);
            //console.log(input.shape)
            this.sumOutput = input.dot(this.weights).add(this.bias);
            this.sumOutput = tf.relu(this.sumOutput);
            return this.sumOutput;
        } else if(this.output) {
            //console.log(input.shape)
            this.sumOutput = input.dot(this.weights).add(this.bias);
            this.sumOutput = tf.softmax(this.sumOutput);
            return this.sumOutput;
        } else {
            //console.log(input.shape)
            this.sumOutput = input.dot(this.weights).add(this.bias);
            this.sumOutput = tf.relu(this.sumOutput);
            return this.sumOutput;
        }
    }

    print() {
        this.weights.print();
    }
}

module.exports = {Layer};
