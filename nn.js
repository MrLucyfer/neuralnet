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

    train(X, y, epochs=0) {
        let output = this.netStruct[0].output(X);
        for(let i = 1; i < this.netStruct.length; i++) {
            this.netStruct[i] = output = this.netStruct[i].output(output);
        }
            this.netStruct[this.netStruct.length - 1].print()
    }

    predict(X) {

    } 

    printLayers() {
        for(let layer of this.netStruct) {
            console.log("Shape of the layer:", layer.weights.shape)
            layer.print();
        }
    } 
}