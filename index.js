const nn = new NeuralNet(4, 10);
const X = tf.data.array([[0, 0], [0,1], [1,0], [1,1]]);
const y = tf.data.array([[1, 0], [0,1], [0,1], [0,1]]);

const dataset = tf.data.zip({X, y});

nn.add(new Layer(2,'relu', true));
nn.add(new Layer(4));
nn.add(new Layer(6));
nn.add(new Layer(2, 'softmax'))

nn.compile();
//nn.printLayers();
nn.train(dataset, 10);