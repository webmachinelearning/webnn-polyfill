import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../execution_context';
import {Operand} from '../operand';
import {Operation} from '../operation';
import * as utils from '../utils';

export class Reshape extends Operation {
  private newShape_: number[];

  constructor(input: Operand, newShape: number[]) {
    super([input]);
    utils.assert(
        utils.isNumberArray(newShape), 'The newShape parameter is invalid.');
    this.newShape_ = newShape;
  }

  run(context: ExecutionContext): tf.Tensor {
    const input: tf.Tensor = this.getTensor(this.inputs[0], context);
    return tf.reshape(input, this.newShape_);
  }
}