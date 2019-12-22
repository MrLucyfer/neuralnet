class Layer {
    constructor(shape, input = false) {
        this.input = input;
        this.shape = shape
        this.weights;
        this.bias = tf.randomUniform([1, this.shape], -1, 1);
        this.output;
    }

    init(inputShape) {
        if(!this.input) {
            this.weights = tf.randomUniform([inputShape, this.shape], -1, 1);
        } else {
            this.weights = tf.randomUniform([this.shape, this.shape], -1, 1);
        }
    }

    output(input) {
        if(this.input == true) {
            input = tf.tensor1d(input);
            let output = tf.dot(input, this.weights)
            output = tf.relu(tf.add(output, this.bias));
            return output
        } else {
            let output = tf.dot(input, this.weights)
            output = tf.relu(tf.add(output, this.bias));
            return output
        }
        
    }

    print() {
        this.weights.print();
    }
}