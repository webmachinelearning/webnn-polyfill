import * as tf from '@tensorflow/tfjs-core';

import {MLOperand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Slice extends SingleOutputOperation {
  private input_: MLOperand;
  private starts_: number[];
  private sizes_: number[];
  private axes_?: number[];
  private needCheckOutputShape_ = true;

  constructor(
      input: MLOperand, starts: number[], sizes: number[], axes?: number[]) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    utils.assert(
        utils.isIntegerArray(starts), 'The starts parameter is invalid.');
    this.starts_ = starts;
    utils.assert(
        utils.isIntegerArray(sizes) && sizes.every(v => v > 0 || v === -1),
        'The sizes parameter is invalid.');
    this.sizes_ = sizes;
    utils.assert(
        sizes.length === sizes.length,
        'The length of sizes is not equal to the length of sizes.))');
    utils.assert(
        axes === undefined || utils.isUnsignedIntegerArray(axes),
        'The axes parameter is invalid.');
    if (axes !== undefined) {
      utils.assert(
          sizes.length === axes.length, 'The length of axes is invalid.))');
    }
    this.axes_ = axes;
  }

  inputs(): MLOperand[] {
    return [this.input_];
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    const input: tf.Tensor4D = inputTensors.get(this.input_) as tf.Tensor4D;
    const rank = input.shape.length;
    if (this.axes_ === undefined) {
      // assume axes is [0, 1,...r-1] if it is not defined.
      this.axes_ = [];
      for (let i = 0; i < rank; ++i) {
        this.axes_.push(i);
      }
    }
    utils.assert(
        this.axes_.every(axis => axis < rank && axis >= 0),
        `The axes must be in range [0, ${rank}).`);
    utils.assert(
        this.starts_.length === this.axes_.length,
        'The length of starts is invalid.');
    utils.assert(
        this.sizes_.length === this.axes_.length,
        'The length of sizes is invalid.');

    const begin: number[] = new Array(this.axes_.length).fill(0);
    const size: number[] = new Array(this.axes_.length).fill(-1);
    for (let i = 0; i < this.axes_.length; ++i) {
      const axis = this.axes_[i];
      begin[axis] = this.starts_[i] >= 0 ? this.starts_[i] :
          this.starts_[i] + input.shape[axis];
      size[axis] = this.sizes_[i];
    }
    const output = tf.slice(input, begin, size);
    if (this.needCheckOutputShape_) {
      const outputShape = input.shape.slice();
      for (let i = 0; i < this.axes_.length; ++i) {
        const axis = this.axes_[i];
        const size = input.shape[axis];
        const start = this.starts_[i];
        const sliceSize = this.sizes_[i];
        if (sliceSize >= 0) {
          outputShape[axis] = sliceSize;
        } else {
          outputShape[axis] = start >= 0 ? size - start : -start;
        }
      }
      utils.checkShape(output.shape, outputShape);
    }
    return output;
  }
}