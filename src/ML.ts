import {ML as MLImpl} from './MLImpl';
import {NeuralNetworkContext} from './nn/NeuralNetworkContext';

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