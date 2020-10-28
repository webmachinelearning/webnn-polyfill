import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../compilation';
import {Operand} from '../operand';
import {Operation} from '../operation';
import * as utils from '../utils';

export class Squeeze extends Operation {
  private axes_: number[];

  constructor(input: Operand, axes?: number[]) {
    super([input]);
    if (axes !== undefined) {
      utils.assert(
          utils.isIntegerArray(axes) && axes.length !== 0,
          'The axes parameter is invalid.');
    }
    this.axes_ = axes;
  }

  run(context: ExecutionContext): tf.Tensor {
    const input: tf.Tensor = this.getTensor(this.inputs[0], context);
    return tf.squeeze(input, this.axes_);
  }
}
