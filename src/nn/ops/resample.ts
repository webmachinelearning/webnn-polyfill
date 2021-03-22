import * as tf from '@tensorflow/tfjs-core';

import {InterpolationMode, ResampleOptions} from '../model_builder';
import {Operand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Resample extends SingleOutputOperation {
  private input_: Operand;
  private mode_: InterpolationMode = InterpolationMode['nearest-neighbor'];
  private scales_: [number, number, number, number];
  private sizes_: [number, number, number, number];

  constructor(input: Operand, options: ResampleOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    if (options.scales !== undefined) {
      const array = options.scales;
      utils.assert(
          array instanceof Array && array.every(v => typeof v === 'number') &&
              array.length === 4,
          'The scales parameter is invalid.');
      this.scales_ = options.scales;
    }
    if (options.sizes !== undefined) {
      utils.assert(
          utils.isIntegerArray(options.sizes) && options.sizes.length === 4,
          'The sizes parameter is invalid.');
      this.sizes_ = options.sizes;
      this.scales_ = undefined;
    }
    utils.assert(
        this.scales_ !== undefined || this.sizes_ !== undefined,
        'The scales or sizes parameter is not provied.');
    if (options.mode !== undefined) {
      utils.assert(
          options.mode in InterpolationMode, 'The mode parameter is invalid.');
      this.mode_ = options.mode;
    }
  }

  inputs(): Operand[] {
    return [this.input_];
  }

  run(inputTensors: Map<Operand, tf.Tensor>): tf.Tensor {
    let input: tf.Tensor4D = inputTensors.get(this.input_) as tf.Tensor4D;
    utils.assert(input.rank === 4, 'The input tensor is not 4-D.');
    const sizes: [number, number] = [0, 0];
    let transposed = false;
    if (this.sizes_ !== undefined) {
      if (this.sizes_[0] === input.shape[0] &&
          this.sizes_[1] === input.shape[1]) {
        sizes[0] = this.sizes_[2];
        sizes[1] = this.sizes_[3];
        // assume nchw -> nhwc
        input = tf.transpose(input, [0, 2, 3, 1]);
        transposed = true;
      } else if (
          this.sizes_[0] === input.shape[0] &&
          this.sizes_[3] === input.shape[3]) {
        // assume nhwc
        sizes[0] = this.sizes_[1];
        sizes[1] = this.sizes_[2];
      } else {
        throw new Error(
            'tf.image.resize doesn\'t support the sizes parameter.');
      }
    } else if (this.scales_ !== undefined) {
      if (this.scales_[0] === 1.0 && this.scales_[1] === 1.0) {
        sizes[0] = Math.floor(input.shape[2] * this.scales_[2]);
        sizes[1] = Math.floor(input.shape[3] * this.scales_[3]);
        // assume nchw -> nhwc
        input = tf.transpose(input, [0, 2, 3, 1]);
        transposed = true;
      } else if (this.scales_[0] === 1.0 && this.scales_[3] === 1.0) {
        // assume nhwc
        sizes[0] = Math.floor(input.shape[1] * this.scales_[1]);
        sizes[1] = Math.floor(input.shape[2] * this.scales_[2]);
      } else {
        throw new Error(
            'tf.image.resize doesn\'t support the scales parameter.');
      }
    }
    let output: tf.Tensor;
    if (this.mode_ === InterpolationMode['nearest-neighbor']) {
      output = tf.image.resizeNearestNeighbor(input, sizes, false, true);
    } else if (this.mode_ === InterpolationMode.linear) {
      output = tf.image.resizeBilinear(input, sizes, false, true);
    }
    if (transposed) {
      // nhwc -> nchw
      output = tf.transpose(output, [0, 3, 1, 2]);
    }
    return output;
  }
}