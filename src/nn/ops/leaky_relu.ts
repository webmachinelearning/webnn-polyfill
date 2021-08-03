import * as tf from '@tensorflow/tfjs-core';

import {MLOperand} from '../operand';
import * as utils from '../utils';

import {UnaryMLOperator} from './unary';

export class LeakyRelu extends UnaryMLOperator {
  private alpha_?: number;

  get alpha(): number {
    return this.alpha_;
  }

  constructor(x: MLOperand, alpha = 0.01) {
    super(x);
    utils.assert(typeof alpha === 'number', 'The alpha parameter is invalid.');
    this.alpha_ = alpha;
  }

  runOp(x: tf.Tensor): tf.Tensor {
    return tf.leakyRelu(x, this.alpha_);
  }
}
