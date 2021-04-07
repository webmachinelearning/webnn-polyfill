import * as tf from '@tensorflow/tfjs-core';

import {MLOperand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Softmax extends SingleOutputOperation {
  private x_: MLOperand;

  constructor(x: MLOperand) {
    super(x.builder);
    utils.validateOperand(x);
    this.x_ = x;
  }

  inputs(): MLOperand[] {
    return [this.x_];
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    const x: tf.Tensor = inputTensors.get(this.x_);
    if (x.rank !== 2) {
      throw new Error('The rank of x parameter should be 2.');
    }
    return tf.softmax(x);
  }
}