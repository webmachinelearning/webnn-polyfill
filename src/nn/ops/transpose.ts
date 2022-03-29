import * as tf from '@tensorflow/tfjs-core';

import {MLOperand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Transpose extends SingleOutputOperation {
  private input_: MLOperand;
  private permutation_?: number[];
  private needCheckOutputShape_ = true;

  constructor(input: MLOperand, permutation?: number[]) {
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

  inputs(): MLOperand[] {
    return [this.input_];
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    const input: tf.Tensor = inputTensors.get(this.input_);
    const output = tf.transpose(input, this.permutation_);
    if (this.needCheckOutputShape_) {
      const inpPermutation = this.permutation_ ??
          new Array(input.rank).fill(0).map((e, i, a) => a.length - i - 1);
      const outputShape = new Array(input.rank).fill(0).map(
          (_, i) => input.shape[inpPermutation[i]]);
      utils.checkShape(output.shape, outputShape);
      this.needCheckOutputShape_ = false;
    }
    return output;
  }
}