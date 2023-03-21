import * as tf from '@tensorflow/tfjs-core';

import {MLOperand} from '../operand';
import {UnaryMLActivation} from './unary';
import * as utils from '../utils';

export class Softplus extends UnaryMLActivation {
  private steepness_?: number;

  get steepness(): number {
    return this.steepness_;
  }

  constructor(x: MLOperand, steepness = 1) {
    super(x);
    utils.assert(
        typeof steepness === 'number', 'The steepness parameter is invalid.');
    this.steepness_ = steepness;
  }

  runOp(x: tf.Tensor): tf.Tensor {
    // The calculation follows the expression
    //     ln(1 + exp(steepness * x)) / steepness
    return tf.div(
      tf.log(tf.add(tf.exp(tf.mul(x, this.steepness_)), 1)),
      this.steepness_
    );
  }
}