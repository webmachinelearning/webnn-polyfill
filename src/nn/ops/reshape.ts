import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../compilation';
import {Operand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Reshape extends SingleOutputOperation {
  private input_: Operand;
  private newShape_: number[];

  constructor(input: Operand, newShape: number[]) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    utils.assert(
        utils.isIntegerArray(newShape) && newShape.length !== 0,
        'The newShape parameter is invalid.');
    this.newShape_ = newShape;
  }

  inputs(): Operand[] {
    return [this.input_];
  }

  run(context: ExecutionContext): tf.Tensor {
    const input: tf.Tensor = context.getTensor(this.input_);
    return tf.reshape(input, this.newShape_);
  }
}