import * as tf from '@tensorflow/tfjs-core';

import {MLOperand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Squeeze extends SingleOutputOperation {
  private input_: MLOperand;
  private axes_?: number[];
  private needCheckOutputShape_ = true;

  constructor(input: MLOperand, axes?: number[]) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    if (axes !== undefined) {
      utils.assert(
          utils.isIntegerArray(axes) && axes.length !== 0,
          'The axes parameter is invalid.');
    }
    this.axes_ = axes;
  }

  inputs(): MLOperand[] {
    return [this.input_];
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    const input: tf.Tensor = inputTensors.get(this.input_);
    const output = tf.squeeze(input, this.axes_);
    if (this.needCheckOutputShape_) {
      const inpAxes = this.axes_ ?? [...Array(input.rank).keys()];
      const outputShape = input.shape.filter(
          (dim, axis) => !(dim === 1 && inpAxes.indexOf(axis) !== -1));
      utils.checkShape(output.shape, outputShape);
      this.needCheckOutputShape_ = false;
    }
    return output;
  }
}
