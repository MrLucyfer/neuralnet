const nn = new NeuralNet(4, 10);
const X = [0, 0];
const y = [0];

nn.add(new Layer(2, true));
nn.add(new Layer(4));
nn.add(new Layer(6));
nn.add(new Layer(1))

nn.compile();
//nn.printLayers();
nn.train(X, y);