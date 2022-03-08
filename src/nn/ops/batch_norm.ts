import * as tf from '@tensorflow/tfjs-core';

import {MLBatchNormalizationOptions} from '../graph_builder';
import {MLOperand, OutputOperand} from '../operand';
import {FusedOperation, MLOperator, SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class BatchNormalization extends SingleOutputOperation implements
    FusedOperation {
  private input_: MLOperand;
  private mean_: MLOperand;
  private variance_: MLOperand;
  private scale_?: MLOperand;
  private bias_?: MLOperand;
  private axis_?: number;
  private epsilon_?: number;
  private activation_?: MLOperator;
  private needCheckOutputShape_ = true;

  constructor(
      input: MLOperand, mean: MLOperand, variance: MLOperand,
      options: MLBatchNormalizationOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    utils.validateOperand(mean);
    this.mean_ = mean;
    utils.validateOperand(variance);
    this.variance_ = variance;
    utils.validateOptionalOperand(options.scale);
    this.scale_ = options.scale;
    utils.validateOptionalOperand(options.bias);
    this.bias_ = options.bias;
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
    this.activation_ = options.activation;
  }

  inputs(): MLOperand[] {
    const inputs: MLOperand[] = [this.input_, this.mean_, this.variance_];
    if (this.scale_) {
      inputs.push(this.scale_);
    }
    if (this.bias_) {
      inputs.push(this.bias_);
    }
    return inputs;
  }

  getFusedOutputs(): OutputOperand[] {
    if (this.activation_) {
      return [this.activation_.apply(this.output)];
    } else {
      return [this.output];
    }
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    const input: tf.Tensor = inputTensors.get(this.input_);
    utils.assert(
        this.axis_ < input.rank && this.axis_ >= -input.rank,
        'The axis parameter is invalid.');
    const axis = this.axis_ >= 0 ? this.axis_ : input.rank + this.axis_;
    const mean: tf.Tensor = inputTensors.get(this.mean_);
    utils.assert(mean.rank === 1, 'The mean operand is not 1-D.');
    const variance: tf.Tensor = inputTensors.get(this.variance_);
    utils.assert(variance.rank === 1, 'The mean operand is not 1-D.');
    let scale: tf.Tensor;
    if (this.scale_) {
      scale = inputTensors.get(this.scale_);
      utils.assert(scale.rank === 1, 'The scale operand is not 1-D.');
    }
    let bias: tf.Tensor;
    if (this.bias_) {
      bias = inputTensors.get(this.bias_);
      utils.assert(bias.rank === 1, 'The bias operand is not 1-D.');
    }
    // tf.batchNorm only computes for the last dimension.
    const permutation = Array.from(Array(input.rank).keys());
    permutation[axis] = input.rank - 1;
    permutation[input.rank - 1] = axis;
    const output: tf.Tensor = tf.transpose(
        tf.batchNorm(
            tf.transpose(input, permutation), mean, variance, bias, scale,
            this.epsilon_),
        permutation);
    if (this.needCheckOutputShape_) {
      // The output shape is the same shape as the input
      utils.checkShape(output.shape, input.shape);
      this.needCheckOutputShape_ = false;
    }
    return output;
  }
}