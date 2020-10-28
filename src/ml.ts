import {NeuralNetworkContext} from './nn/neural_network_context';

/**
 * [ML](https://webmachinelearning.github.io/webnn/#api-ml)
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
