import {ML as MLInterface} from './ML';
import {NeuralNetworkContext} from './nn/NeuralNetworkContextImpl';

export class ML implements MLInterface {
  private nnContext: NeuralNetworkContext = null;

  getNeuralNetworkContext(): NeuralNetworkContext {
    if (!this.nnContext) {
      this.nnContext = new NeuralNetworkContext();
    }
    return this.nnContext;
  }
}