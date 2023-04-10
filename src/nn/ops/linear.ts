import * as tf from '@tensorflow/tfjs-core';

import {MLLinearOptions} from '../graph_builder';
import {MLOperand} from '../operand';
import {UnaryMLActivation} from './unary';
import * as utils from '../utils';


export class Linear extends UnaryMLActivation {
    private alpha_?: number = 1;
    private beta_?: number = 0;

    get alpha(): number {
      return this.alpha_;
    }

    get beta(): number {
      return this.beta_;
    }

    constructor(x: MLOperand, options: MLLinearOptions = {}) {
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
      // alpha * x + beta
      return tf.add(tf.mul(x, this.alpha_), this.beta_);
    }
}