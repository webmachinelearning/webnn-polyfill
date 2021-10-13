import * as tf from '@tensorflow/tfjs-core';

import {MLInterpolationMode, MLResample2dOptions} from '../graph_builder';
import {MLOperand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Resample2d extends SingleOutputOperation {
  private input_: MLOperand;
  private mode_: MLInterpolationMode = MLInterpolationMode['nearest-neighbor'];
  private scales_: [number, number] = [1.0, 1.0];
  private sizes_: [number, number];
  private axes_: [number, number] = [2, 3];

  constructor(input: MLOperand, options: MLResample2dOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    if (options.scales !== undefined) {
      const array = options.scales;
      utils.assert(
          array instanceof Array && array.every(v => typeof v === 'number') &&
              array.length === 2,
          'The scales parameter is invalid.');
      this.scales_ = options.scales;
    }
    if (options.sizes !== undefined) {
      utils.assert(
          utils.isIntegerArray(options.sizes) && options.sizes.length === 2,
          'The sizes parameter is invalid.');
      this.sizes_ = options.sizes;
    }
    if (options.axes !== undefined) {
      utils.assert(
          utils.isIntegerArray(options.axes) && options.axes.length === 2 &&
              utils.isValidResample2dAxes(options.axes),
          'The axes parameter is invalid.');
      this.axes_ = options.axes;
    }
    utils.assert(
        this.scales_ !== undefined || this.sizes_ !== undefined,
        'The scales or sizes parameter is not provied.');
    if (options.mode !== undefined) {
      utils.assert(
          options.mode in MLInterpolationMode,
          'The mode parameter is invalid.');
      this.mode_ = options.mode;
    }
  }

  inputs(): MLOperand[] {
    return [this.input_];
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    let input: tf.Tensor4D = inputTensors.get(this.input_) as tf.Tensor4D;
    utils.assert(input.rank === 4, 'The input tensor is not 4-D.');
    const sizes: [number, number] = [0, 0];
    if (this.sizes_ !== undefined) {
        // ignore scales
        sizes[0] = this.sizes_[0];
        sizes[1] = this.sizes_[1];
    } else if (this.scales_ !== undefined) {
        sizes[0] = Math.floor(input.shape[this.axes_[0]] * this.scales_[0]);
        sizes[1] = Math.floor(input.shape[this.axes_[1]] * this.scales_[1]);
    }
    if (this.axes_[0] === 0) {
      // hwnc -> nhwc
      input = tf.transpose(input, [2, 0, 1, 3]);
    } else if (this.axes_[0] === 2) {
      // nchw -> nhwc
      input = tf.transpose(input, [0, 2, 3, 1]);
    }
    let output: tf.Tensor;
    if (this.mode_ === MLInterpolationMode['nearest-neighbor']) {
      output = tf.image.resizeNearestNeighbor(input, sizes, false, true);
    } else if (this.mode_ === MLInterpolationMode.linear) {
      output = tf.image.resizeBilinear(input, sizes, false, true);
    }
    if (this.axes_[0] === 0) {
      // nhwc -> hwnc
      output = tf.transpose(output, [1, 2, 0, 3]);
    } else if (this.axes_[0] === 2) {
      // nhwc -> nchw
      output = tf.transpose(output, [0, 3, 1, 2]);
    }
    return output;
  }
}