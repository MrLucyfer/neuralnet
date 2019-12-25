const nn = require('../src/nn.js')
const mnist = require('./mnist.js')
function oneHot(x) {
    let arr = new Array(10).fill(0);
    arr[x] = 1;
    //console.log(arr)
    return arr;
}

function normalize(images) {
    norm_img = [];
    for(let i = 0; i < 20000; i++) {
        norm_img.push([])
        images[i].forEach((x) => norm_img[i].push(x/255))
    }
    return norm_img;
}

async function setup() {

    await mnist.loadMNIST((data) => {
        mnist = data;
        console.log("Dataset loaded")
    })
    
    //Preparamos el dataset
    mnist.train_labels_onehot = new Array()
    for(let i = 0; i < mnist.train_labels.length; i++) {
        mnist.train_labels_onehot.push(oneHot(mnist.train_labels[i]));
    }
    mnist.train_norm_img = normalize(mnist.train_images);

    //Creamos el dataset
    const X = tf.data.array(mnist.train_norm_img);
    const y = tf.data.array(mnist.train_labels_onehot);
    const dataset = tf.data.zip({X, y});

    //Creamos la NN
    nn = new NeuralNet();

    //Añadimos capas a la nn
    nn.add(new Layer([784, 32], 'relu', true));
    nn.add(new Layer([32, 10], 'softmax'));

    nn.compile(); 
    //Entrenamos la nn
    nn.train(dataset, 10)
}

setup()