import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../compilation';
import {Operand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Transpose extends SingleOutputOperation {
  private input_: Operand;
  private permutation_?: number[];

  constructor(input: Operand, permutation?: number[]) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    if (permutation !== undefined) {
      utils.assert(
          utils.isIntegerArray(permutation) && permutation.length !== 0,
          'The permutation parameter is invalid.');
    }
    this.permutation_ = permutation;
  }

  inputs(): Operand[] {
    return [this.input_];
  }

  run(context: ExecutionContext): tf.Tensor {
    const input: tf.Tensor = context.getTensor(this.input_);
    return tf.transpose(input, this.permutation_);
  }
}