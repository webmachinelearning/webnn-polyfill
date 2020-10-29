import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../compilation';
import {BatchNormalizationOptions} from '../model_builder';
import {Operand} from '../operand';
import {Operation} from '../operation';
import * as utils from '../utils';

export class BatchNormalization extends Operation {
  private axis_: number;
  private epsilon_: number;

  constructor(
      input: Operand, mean: Operand, variance: Operand,
      options: BatchNormalizationOptions = {}) {
    const inputs: Operand[] = [input, mean, variance];
    if (options.scale !== undefined) {
      inputs.push(options.scale);
    }
    if (options.bias !== undefined) {
      inputs.push(options.bias);
    }
    super(inputs);
    if (options.axis !== undefined) {
      const axis = options.axis;
      utils.assert(utils.isInteger(axis), 'The axis parameter is invalid.');
      this.axis_ = axis;
    } else {
      this.axis_ = 1;
    }
    if (options.epsilon !== undefined) {
      const epsilon = options.epsilon;
      utils.assert(
          typeof epsilon === 'number', 'The epsilon parameter is invalid');
      this.epsilon_ = epsilon;
    } else {
      this.epsilon_ = 1e-5;
    }
  }

  run(context: ExecutionContext): tf.Tensor {
    const input: tf.Tensor = this.getTensor(this.inputs[0], context);
    utils.assert(
        this.axis_ < input.rank && this.axis_ >= -input.rank,
        'The axis parameter is invalid.');
    const axis = this.axis_ >= 0 ? this.axis_ : input.rank + this.axis_;
    const mean: tf.Tensor = this.getTensor(this.inputs[1], context);
    utils.assert(mean.rank === 1, 'The mean operand is not 1-D.');
    const variance: tf.Tensor = this.getTensor(this.inputs[2], context);
    utils.assert(variance.rank === 1, 'The mean operand is not 1-D.');
    let scale: tf.Tensor;
    if (this.inputs.length > 3) {
      scale = this.getTensor(this.inputs[3], context);
      utils.assert(scale.rank === 1, 'The scale operand is not 1-D.');
    }
    let bias: tf.Tensor;
    if (this.inputs.length > 4) {
      bias = this.getTensor(this.inputs[4], context);
      utils.assert(bias.rank === 1, 'The bias operand is not 1-D.');
    }
    // tf.batchNorm only computes for the last dimension.
    const permutation = Array.from(Array(input.rank).keys());
    permutation[axis] = input.rank - 1;
    permutation[input.rank - 1] = axis;
    return tf
        .batchNorm(
            tf.transpose(input, permutation), mean, variance, bias, scale,
            this.epsilon_)
        .transpose(permutation);
  }
}