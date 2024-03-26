import * as tf from '@tensorflow/tfjs-core';

import {MLHardSigmoidOptions} from '../graph_builder';
import {MLOperand} from '../operand';
import {UnaryMLActivation} from './unary';
import * as utils from '../utils';


export class HardSigmoid extends UnaryMLActivation {
    private alpha_?: number = 0.2;
    private beta_?: number = 0.5;

    get alpha(): number {
      return this.alpha_;
    }

    get beta(): number {
      return this.beta_;
    }

    constructor(x: MLOperand, options: MLHardSigmoidOptions = {}) {
      super(x);
      utils.validateOperand(x);

      if (options.alpha !== undefined) {
        const alpha = options.alpha;
        utils.assert(
            typeof alpha === 'number', 'The alpha parameter is invalid');
        this.alpha_ = alpha;
      }

      if (options.beta !== undefined) {
        const beta = options.beta;
        utils.assert(
            typeof beta === 'number', 'The beta parameter is invalid');
        this.beta_ = beta;
      }
    }

    runOp(x: tf.Tensor): tf.Tensor {
      // max(min(alpha * x + beta, 1), 0)
      return tf.maximum(
        tf.minimum(tf.add(tf.mul(this.alpha_, x), this.beta_), 1),
        0
      );
    }
}