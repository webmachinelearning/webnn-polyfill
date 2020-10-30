import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../compilation';
import {Operand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Softmax extends SingleOutputOperation {
  private x_: Operand;

  constructor(x: Operand) {
    super(x.builder);
    utils.validateOperand(x);
    this.x_ = x;
  }

  inputs(): Operand[] {
    return [this.x_];
  }

  run(context: ExecutionContext): tf.Tensor {
    const x: tf.Tensor = context.getTensor(this.x_);
    if (x.rank !== 2) {
      throw new Error('The rank of x parameter should be 2.');
    }
    return tf.softmax(x);
  }
}