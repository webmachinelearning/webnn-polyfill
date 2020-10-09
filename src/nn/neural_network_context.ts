import {ModelBuilder} from './model_builder_impl';
import {NeuralNetworkContext as NeuralNetworkContextImpl} from './neural_network_context_impl';


/**
 * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext)
 */
export interface NeuralNetworkContext {
  /** */
  createModelBuilder(): ModelBuilder;
}

interface NeuralNetworkContextConstructor {
  new(): NeuralNetworkContext;
}
// eslint-disable-next-line no-redeclare
export const NeuralNetworkContext: NeuralNetworkContextConstructor =
    NeuralNetworkContextImpl;
