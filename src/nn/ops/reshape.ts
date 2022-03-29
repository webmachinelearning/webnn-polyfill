import * as tf from '@tensorflow/tfjs-core';

import {MLOperand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Reshape extends SingleOutputOperation {
  private input_: MLOperand;
  private newShape_: number[];
  private needCheckOutputShape_ = true;

  constructor(input: MLOperand, newShape: number[]) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    utils.assert(
        utils.isIntegerArray(newShape) && newShape.length !== 0,
        'The newShape parameter is invalid.');
    this.newShape_ = newShape;
  }

  inputs(): MLOperand[] {
    return [this.input_];
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    const input: tf.Tensor = inputTensors.get(this.input_);
    const output: tf.Tensor = tf.reshape(input, this.newShape_);
    if (this.needCheckOutputShape_) {
      // The output shape is specified by the newShape argument
      const outputShape: number[] = this.newShape_.slice();
      if (outputShape.indexOf(-1) !== -1) {
        // Only one component of newShape can be the special value of -1.
        // The values of the output tensor are the same as values of the input
        // tensor.
        outputShape[outputShape.indexOf(-1)] =
            utils.product(input.shape) / utils.product(outputShape) * -1;
      }
      utils.checkShape(output.shape, outputShape);
      this.needCheckOutputShape_ = false;
    }
    return output;
  }
}