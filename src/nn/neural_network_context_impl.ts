import {ModelBuilder} from './model_builder_impl';
import {NeuralNetworkContext as NeuralNetworkContextInterface} from './neural_network_context';

export class NeuralNetworkContext implements NeuralNetworkContextInterface {
  createModelBuilder(): ModelBuilder {
    return new ModelBuilder();
  }
}
