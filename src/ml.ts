import {NeuralNetworkContext} from './nn/neural_network_context';

/**
 * [API spec](https://webmachinelearning.github.io/webnn/#ml)
 */
export class ML {
  private nnContext: NeuralNetworkContext = null;

  /** @ignore */
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  constructor() {}

  getNeuralNetworkContext(): NeuralNetworkContext {
    if (!this.nnContext) {
      this.nnContext = new NeuralNetworkContext();
    }
    return this.nnContext;
  }
}
