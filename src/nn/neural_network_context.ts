import {ModelBuilder} from './model_builder';

/**
 * [API spec](https://webmachinelearning.github.io/webnn/#neuralnetworkcontext)
 */
export class NeuralNetworkContext {
  /** */
  createModelBuilder(): ModelBuilder {
    return new ModelBuilder();
  }
}
