import {NeuralNetworkContext} from './nn/NeuralNetworkContext';

/**
 * Implements the [ML](https://webmachinelearning.github.io/webnn/#api-ml)
 * interface.
 */
export class ML {
  private nnContext: NeuralNetworkContext = null;

  /** */
  getNeuralNetworkContext(): NeuralNetworkContext {
    if (!this.nnContext) {
      this.nnContext = new NeuralNetworkContext();
    }
    return this.nnContext;
  }
}