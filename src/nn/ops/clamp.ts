import * as tf from '@tensorflow/tfjs-core';

import {MLClampOptions} from '../graph_builder';
import {MLOperand} from '../operand';
import {UnaryMLActivation} from './unary';
import * as utils from '../utils';

export class Clamp extends UnaryMLActivation {
  private minValue_?: number;
  private maxValue_?: number;

  get minValue(): number {
    return this.minValue_ !== undefined ? this.minValue_ : -Infinity;
  }
  get maxValue(): number {
    return this.maxValue_ !== undefined ? this.maxValue_ : +Infinity;
  }

  constructor(x: MLOperand, options: MLClampOptions = {}) {
    if (x !== undefined) {
      super(x);
      utils.validateOperand(x);
      this.x_ = x;
    } else {
      super(undefined);
      this.x_ = undefined;
    }

    if (options.minValue !== undefined) {
      const minValue = options.minValue;
      utils.assert(
          typeof minValue === 'number', 'The minValue parameter is invalid');
      this.minValue_ = minValue;
    }

    if (options.maxValue !== undefined) {
      const maxValue = options.maxValue;
      utils.assert(
          typeof maxValue === 'number', 'The maxValue parameter is invalid');
      this.maxValue_ = maxValue;
    }
  }

  runOp(x: tf.Tensor): tf.Tensor {
    if (this.minValue_ !== undefined && this.maxValue_ !== undefined) {
      return tf.clipByValue(x, this.minValue_, this.maxValue_);
    } else if (this.minValue_ !== undefined && this.maxValue_ === undefined) {
      return tf.maximum(x, this.minValue_);
    } else if (this.minValue_ === undefined && this.maxValue_ !== undefined) {
      return tf.minimum(x, this.maxValue_);
    } else {
      return tf.clone(x);
    }
  }
}