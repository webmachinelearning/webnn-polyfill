import {ModelBuilder} from './model_builder';

/**
 * [NeuralNetworkContext](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext)
 */
export class NeuralNetworkContext {
  /** */
  createModelBuilder(): ModelBuilder {
    return new ModelBuilder();
  }
}
