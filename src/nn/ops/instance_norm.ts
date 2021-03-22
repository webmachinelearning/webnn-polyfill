import * as tf from '@tensorflow/tfjs-core';

import {InputOperandLayout, InstanceNormalizationOptions} from '../model_builder';
import {Operand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class InstanceNormalization extends SingleOutputOperation {
  private input_: Operand;
  private scale_?: Operand;
  private bias_?: Operand;
  private epsilon_?: number;
  private layout_: InputOperandLayout;

  constructor(input: Operand, options: InstanceNormalizationOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    utils.validateOptionalOperand(options.scale);
    this.scale_ = options.scale;
    utils.validateOptionalOperand(options.bias);
    this.bias_ = options.bias;
    if (options.epsilon !== undefined) {
      const epsilon = options.epsilon;
      utils.assert(
          typeof epsilon === 'number', 'The epsilon parameter is invalid');
      this.epsilon_ = epsilon;
    } else {
      this.epsilon_ = 1e-5;
    }
    if (options.layout !== undefined) {
      utils.assert(
          options.layout in InputOperandLayout,
          'The layout parameter is invalid.');
      this.layout_ = options.layout;
    } else {
      this.layout_ = InputOperandLayout.nchw;
    }
  }

  inputs(): Operand[] {
    const inputs: Operand[] = [this.input_];
    if (this.scale_) {
      inputs.push(this.scale_);
    }
    if (this.bias_) {
      inputs.push(this.bias_);
    }
    return inputs;
  }

  run(inputTensors: Map<Operand, tf.Tensor>): tf.Tensor {
    const input: tf.Tensor = inputTensors.get(this.input_);
    utils.assert(input.rank === 4, 'The input operand is not 4-D.');
    let axes = [2, 3];
    let shape = [1, -1, 1, 1];
    let inputChannels = input.shape[1];
    if (this.layout_ === InputOperandLayout.nhwc) {
      axes = [1, 2];
      shape = [1, 1, 1, -1];
      inputChannels = input.shape[3];
    }
    let scale: tf.Tensor;
    if (this.scale_) {
      scale = inputTensors.get(this.scale_);
      utils.assert(scale.rank === 1, 'The scale operand is not 1-D.');
      utils.assert(
          scale.shape[0] === inputChannels,
          'The length of scale is not equal to the size of the feature ' +
              'dimension of the input.');
    } else {
    }
    let bias: tf.Tensor;
    if (this.bias_) {
      bias = inputTensors.get(this.bias_);
      utils.assert(bias.rank === 1, 'The bias operand is not 1-D.');
      utils.assert(
          bias.shape[0] === inputChannels,
          'The length of bias is not equal to the size of the feature ' +
              'dimension of the input.');
    }

    const mean = tf.mean(input, axes, true);
    const variance = tf.mean(tf.pow(tf.sub(input, mean), 2), axes, true);
    const norm = tf.div(
        tf.sub(input, mean), tf.pow(tf.add(variance, this.epsilon_), 0.5));
    const scaled = scale ? tf.mul(tf.reshape(scale, shape), norm) : norm;
    return bias ? tf.add(tf.reshape(bias, shape), scaled) : scaled;
  }
}