import * as tf from '@tensorflow/tfjs-core';

import {MLOperand, OutputOperand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Transpose extends SingleOutputOperation {
  private input_: MLOperand;
  private permutation_?: number[];
  private needCheckOutputShape_ = true;
  private outputShape_: number[];

  constructor(input: MLOperand, permutation?: number[]) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    if (permutation !== undefined) {
      utils.assert(
          utils.isUnsignedIntegerArray(permutation) && permutation.length !== 0,
          'The permutation parameter is invalid.');
    }
    this.permutation_ = permutation;
    this.createOutput();
  }

  inputs(): MLOperand[] {
    return [this.input_];
  }

  createOutput(): void {
    const inputRank = this.input_.rank();
    const inpPermutation = this.permutation_ ??
        new Array(inputRank).fill(0).map((e, i, a) => a.length - i - 1);
    this.outputShape_ = new Array(inputRank).fill(0)
        .map((_, i) => this.input_.shape()[inpPermutation[i]]);
    this.outputs_.push(new OutputOperand(this, 
      {dataType: this.input_.dataType(), dimensions: this.outputShape_}));
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    const input: tf.Tensor = inputTensors.get(this.input_);
    const output = tf.transpose(input, this.permutation_);
    if (this.needCheckOutputShape_) {
      utils.checkShape(output.shape, this.outputShape_);
      this.needCheckOutputShape_ = false;
    }
    return output;
  }
}