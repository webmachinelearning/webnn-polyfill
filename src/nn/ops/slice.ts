import * as tf from '@tensorflow/tfjs-core';

import {MLOperand, OutputOperand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Slice extends SingleOutputOperation {
  private input_: MLOperand;
  private starts_: number[];
  private sizes_: number[];
  private needCheckOutputShape_ = true;

  constructor(
      input: MLOperand, starts: number[], sizes: number[]) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    utils.assert(
        utils.isUnsignedIntegerArray(starts),
        'The starts parameter is invalid.');
    this.starts_ = starts;
    utils.assert(
        utils.isUnsignedIntegerArray(sizes) && sizes.every(v => v > 0),
        'The sizes parameter is invalid.');
    this.sizes_ = sizes;
    this.createOutput();
  }

  inputs(): MLOperand[] {
    return [this.input_];
  }

  createOutput(): void {
    this.outputs_.push(new OutputOperand(this,
      {dataType: this.input_.dataType(), dimensions: this.sizes_})); 
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    const input: tf.Tensor4D = inputTensors.get(this.input_) as tf.Tensor4D;
    const rank = input.shape.length;
    utils.assert(
        this.starts_.length === rank, 'The length of starts is invalid.');
    utils.assert(
        this.sizes_.length === rank, 'The length of sizes is invalid.');
    for (let dimension = 0; dimension < rank; dimension++) {
      const inputSize = input.shape[dimension];
      const begin = this.starts_[dimension];
      utils.assert(
          begin < inputSize, 'The length of starts is invalid.');
      utils.assert(
          begin + this.sizes_[dimension] <= inputSize,
          'The sizes parameter is invalid.');
    }
    const output = tf.slice(input, this.starts_, this.sizes_);
    if (this.needCheckOutputShape_) {
      utils.checkShape(output.shape, this.sizes_);
    }
    return output;
  }
}