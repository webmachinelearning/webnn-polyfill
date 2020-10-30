import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../compilation';
import {ClampOptions} from '../model_builder';
import {Operand} from '../operand';
import {Operation} from '../operation';

export class Clamp extends Operation {
  private minValue_?: Operand;
  private maxValue_?: Operand;

  constructor(x: Operand, options: ClampOptions = {}) {
    const inputs: Operand[] = [];
    if (options.minValue !== undefined) {
      inputs.push(options.minValue);
    }
    if (options.maxValue !== undefined) {
      inputs.push(options.maxValue);
    }
    super(inputs);
    // Add the references of minValue and maxValue.
    this.minValue_ = options.minValue;
    this.maxValue_ = options.maxValue;
  }

  run(context: ExecutionContext): tf.Tensor {
    const x: tf.Tensor = this.getTensor(this.inputs[0], context);
    if (this.minValue_ === undefined) {
      if (this.maxValue_ === undefined) {
        return x;
      } else {
        const max: tf.Tensor = this.getTensor(this.maxValue_, context);
        return tf.minimum(x, max);
      }
    } else {
      const min: tf.Tensor = this.getTensor(this.minValue_, context);
      if (this.maxValue_ === undefined) {
        return tf.maximum(x, min);
      } else {
        const max: tf.Tensor = this.getTensor(this.maxValue_, context);
        return tf.minimum(tf.maximum(x, min), max);
      }
    }
  }
}