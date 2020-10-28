import {ModelBuilder} from './model_builder';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext)
 */
export class NeuralNetworkContext {
  createModelBuilder(): ModelBuilder {
    return new ModelBuilder();
  }
}
