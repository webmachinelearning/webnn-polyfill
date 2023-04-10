import * as tf from '@tensorflow/tfjs-core';

import {MLOperand} from '../operand';
import {UnaryMLActivation} from './unary';
import * as utils from '../utils';

export class Elu extends UnaryMLActivation {
  private alpha_?: number;

  get alpha(): number {
    return this.alpha_;
  }

  constructor(x: MLOperand, alpha = 1) {
    super(x);
    utils.assert(typeof alpha === 'number', 'The alpha parameter is invalid.');
    this.alpha_ = alpha;
  }

  runOp(x: tf.Tensor): tf.Tensor {
    // Since there's an issue of elu https://github.com/tensorflow/tfjs/issues/7496
    // we can't directly invoke current tf.selu(x)

    // The calculation follows the expression
    //     max(0, x) + alpha * (exp(min(0, x)) - 1)
    return tf.add(
      tf.maximum(0, x),
      tf.mul(this.alpha_, tf.sub(tf.exp(tf.minimum(0, x)), 1))
    );
  }
}