import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';

import * as tf from '@tensorflow/tfjs-core';

import {ModelBuilder} from './model_builder';

/**
 * [API spec](https://webmachinelearning.github.io/webnn/#neuralnetworkcontext)
 */
export class NeuralNetworkContext {
  /** */
  createModelBuilder(): ModelBuilder {
    return new ModelBuilder();
  }

  // Expose tf.js for backend debugging.
  get tf(): unknown {
    return tf;
  }
}
