const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const {NeuralNet} = require('./nn.js');
const {Layer} = require('./layer.js');
const nn = new NeuralNet();

nn.add(new Layer([2, 4], 'relu', true));
nn.add(new Layer([4, 8]));
nn.add(new Layer([8, 2], 'softmax'));

nn.compile();
const pred = nn.feedForward([1,1])
pred.print();
