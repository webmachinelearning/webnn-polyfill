import * as tf from '@tensorflow/tfjs-core';

import {MLOperand, OutputOperand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Softmax extends SingleOutputOperation {
  private x_: MLOperand;
  private needCheckOutputShape_ = true;

  constructor(x: MLOperand) {
    super(x.builder);
    utils.validateOperand(x);
    this.x_ = x;
    this.createOutput();
  }

  inputs(): MLOperand[] {
    return [this.x_];
  }

  createOutput(): void {
    this.outputs_.push(new OutputOperand(this, this.x_.desc));
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    const x: tf.Tensor = inputTensors.get(this.x_);
    if (x.rank !== 2) {
      throw new Error('The rank of x parameter should be 2.');
    }
    const output: tf.Tensor = tf.softmax(x);
    if (this.needCheckOutputShape_) {
      // The output shape is the same shape as the input
      utils.checkShape(output.shape, x.shape);
      this.needCheckOutputShape_ = false;
    }
    return output;
  }
}