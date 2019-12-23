class Layer {
    constructor(shape, activation = 'relu', input = false) {
        this.input = input;
        this.shape = shape
        this.weights;
        this.activation = activation;
        this.bias = tf.variable(tf.randomUniform([1, this.shape], -1, 1));
        this.output;
        this.variables;
    }

    init(inputShape) {
        if(!this.input) {
            this.weights = tf.variable(tf.randomUniform([inputShape, this.shape], -1, 1),true);
        } else {
            this.weights = tf.variable(tf.randomUniform([this.shape, this.shape], -1, 1),true);
        }

        this.variables = [this.weights, this.bias];
    }

    weightSum(input) {
        if(this.input == true) {
            input = tf.tensor1d(input);
            let output = tf.dot(input, this.weights);
            this.output = tf.relu(tf.add(output, this.bias));
            return output;
        } else if(this.activation == 'softmax') {
            let output = tf.dot(input, this.weights)
            this.output = tf.softmax(tf.add(output, this.bias));
            return output;
        } else {
            let output = tf.dot(input, this.weights)
            this.output = tf.relu(tf.add(output, this.bias));
            return output;
        }
    }

    print() {
        this.weights.print();
    }
}