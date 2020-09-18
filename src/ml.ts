import {ML as MLImpl} from './ml_impl';
import {NeuralNetworkContext} from './nn/neural_network_context';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#api-ml)
 */
export interface ML {
  /** */
  getNeuralNetworkContext(): NeuralNetworkContext;
}

interface MLConstructor {
  new(): ML;
}
// eslint-disable-next-line no-redeclare
export const ML: MLConstructor = MLImpl;