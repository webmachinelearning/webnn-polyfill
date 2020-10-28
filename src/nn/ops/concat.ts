import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../compilation';
import {Operand} from '../operand';
import {Operation} from '../operation';
import * as utils from '../utils';

export class Concat extends Operation {
  private axis_: number;

  constructor(inputs: Operand[], axis: number) {
    super(inputs);
    utils.assert(utils.isInteger(axis), 'The axis parameter is invalid.');
    this.axis_ = axis;
  }

  run(context: ExecutionContext): tf.Tensor {
    const inputs: tf.Tensor[] = [];
    for (const input of this.inputs) {
      inputs.push(this.getTensor(input, context));
    }
    return tf.concat(inputs, this.axis_);
  }
}
