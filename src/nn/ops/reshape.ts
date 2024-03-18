import * as tf from '@tensorflow/tfjs-core';

import {MLOperand, OutputOperand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Reshape extends SingleOutputOperation {
  private input_: MLOperand;
  private newShape_: Array<(number | null)>;
  private needCheckOutputShape_ = true;

  constructor(input: MLOperand, newShape: Array<(number | null)>) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    utils.assert(
        utils.isPositiveIntegerOrNullArray(newShape) && newShape.length !== 0,
        'The newShape parameter is invalid.');
    this.newShape_ = newShape;
    this.createOutput();
  }

  inputs(): MLOperand[] {
    return [this.input_];
  }

  createOutput(): void {
    this.outputs_.push(new OutputOperand(this,
      {dataType: this.input_.dataType(), dimensions: this.newShape_}));
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    const input: tf.Tensor = inputTensors.get(this.input_);
    const outputShape: Array<(number | null)> = this.newShape_.slice();
    const nullPosition = outputShape.indexOf(null);
    if (nullPosition !== -1) {
      outputShape[nullPosition] = -1;
      outputShape[nullPosition] =
          utils.product(input.shape) / utils.product(outputShape) * -1;
    }
    const output: tf.Tensor = tf.reshape(input, outputShape);
    if (this.needCheckOutputShape_) {
      utils.checkShape(output.shape, outputShape);
      this.needCheckOutputShape_ = false;
    }
    return output;
  }
}