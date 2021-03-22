import * as tf from '@tensorflow/tfjs-core';

import {Operand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class LeakyRelu extends SingleOutputOperation {
  private x_: Operand;
  private alpha_?: number;

  constructor(x: Operand, alpha = 0.01) {
    super(x.builder);
    utils.validateOperand(x);
    this.x_ = x;
    utils.assert(typeof alpha === 'number', 'The alpha parameter is invalid.');
    this.alpha_ = alpha;
  }

  inputs(): Operand[] {
    return [this.x_];
  }

  run(inputTensors: Map<Operand, tf.Tensor>): tf.Tensor {
    const x: tf.Tensor = inputTensors.get(this.x_);
    return tf.leakyRelu(x, this.alpha_);
  }
}
