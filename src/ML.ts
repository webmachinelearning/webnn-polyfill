import { NeuralNetworkContext } from './nn/NeuralNetworkContext'

/**
 * Implements the [ML](https://webmachinelearning.github.io/webnn/#api-ml) interface.
 */
export class ML {
  private nnContext: NeuralNetworkContext = null;

  /**
   * Implements the [getNeuralNetworkContext](https://webmachinelearning.github.io/webnn/#dom-ml-getneuralnetworkcontext) method.
   * 
   * @returns A [[NeuralNetworkContext]] object.
   */
  getNeuralNetworkContext() {
    if (!this.nnContext) {
      this.nnContext = new NeuralNetworkContext();
    }
    return this.nnContext;
  }
}