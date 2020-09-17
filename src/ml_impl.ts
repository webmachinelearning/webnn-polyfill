import {ML as MLInterface} from './ml';
import {NeuralNetworkContext} from './nn/neural_network_context_impl';

export class ML implements MLInterface {
  private nnContext: NeuralNetworkContext = null;

  getNeuralNetworkContext(): NeuralNetworkContext {
    if (!this.nnContext) {
      this.nnContext = new NeuralNetworkContext();
    }
    return this.nnContext;
  }
}