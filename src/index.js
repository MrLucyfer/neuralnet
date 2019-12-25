const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const { NeuralNet } = require('./nn.js');
const { Layer } = require('./layer.js');
const mnist = require('mnist');

const nn = new NeuralNet();

nn.add(new Layer([784, 32], 'relu', true));
nn.add(new Layer([32, 10], 'softmax'));

nn.compile();

function getData() {
    const datasetMnist = mnist.set(20000, 1000);
    const train = datasetMnist.training;
    const test = datasetMnist.test;

    return {
        train,
        test
    }
}

function argMax(array) {
    return array.map((x, i) => [x, i]).reduce((r, a) => (a[0] > r[0] ? a : r))[1];
}
//console.log(getData());
const digits = getData();
async function testNN(test) {
    let solution;
    let rightCount = 0;
    await test.forEachAsync((data) => {
        solution = argMax(data.output)
        const pred = nn.predict(data.input).dataSync();
        if (solution == pred) {
            rightCount++;
        }
    });
    console.log(`Test ended with: ${rightCount}/1000`);
}

async function trainData() {
    let data = tf.data.array(digits.train);
    let dataset = tf.data.zip(data);

    await nn.train(dataset, 5);
    await testNN(dataset);
}

trainData();



// let correct = 0;
// for(let i = 0; i < 4; i++) {

// }
