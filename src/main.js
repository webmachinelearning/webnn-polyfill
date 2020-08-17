import {NeuralNetworkContext} from './nn/NeuralNetworkContext'

class ML {
	constructor() {
    this._nnContext = null;
  }

  getNeuralNetworkContext() {
    if (!this._nnContext) {
      this._nnContext = new NeuralNetworkContext();
    }
    return this._nnContext;
  }
}

if (typeof navigator.ml === 'undefined') {
  navigator.ml = new ML();
}