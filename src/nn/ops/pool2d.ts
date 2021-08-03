import * as tf from '@tensorflow/tfjs-core';
import {ExplicitPadding} from '@tensorflow/tfjs-core/src/ops/conv_util';

import {MLAutoPad, MLInputOperandLayout, MLPooling2dOptions} from '../graph_builder';
import {MLOperand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

type PoolingType = 'avg'|'max';

export abstract class Pool extends SingleOutputOperation {
  protected input_: MLOperand;
  protected windowDimensions_?: [number, number];
  protected padding_?: [number, number, number, number];
  protected strides_?: [number, number];
  protected dilations_?: [number, number];
  protected groups_?: number;
  protected layout_?: MLInputOperandLayout;
  private autoPad_?: MLAutoPad;

  constructor(input: MLOperand, options: MLPooling2dOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    this.initOptions(
        options.windowDimensions, options.padding, options.strides,
        options.dilations, options.layout, options.autoPad);
  }

  private initOptions(
      windowDimensions: [number, number] = [-1, -1],
      padding: [number, number, number, number] = [0, 0, 0, 0],
      strides: [number, number] = [1, 1], dilations: [number, number] = [1, 1],
      layout: MLInputOperandLayout = MLInputOperandLayout.nchw,
      autoPad: MLAutoPad = MLAutoPad.explicit) {
    utils.assert(
        utils.isIntegerArray(windowDimensions) && windowDimensions.length === 2,
        'The padding parameter is invalid.');
    this.windowDimensions_ = windowDimensions;

    utils.assert(
        utils.isIntegerArray(padding) && padding.length === 4,
        'The padding parameter is invalid.');
    this.padding_ = padding;

    utils.assert(
        utils.isIntegerArray(strides) && strides.length === 2,
        'The strides parameter is invalid.');
    this.strides_ = strides;

    utils.assert(
        utils.isIntegerArray(dilations) && dilations.length === 2,
        'The dilations parameter is invalid.');
    this.dilations_ = dilations;

    utils.assert(
        layout in MLInputOperandLayout, 'The layout parameter is invalid.');
    this.layout_ = layout;

    utils.assert(autoPad in MLAutoPad, 'The autoPad parameter is invalid.');
    this.autoPad_ = autoPad;
  }

  inputs(): MLOperand[] {
    return [this.input_];
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    let input: tf.Tensor4D = inputTensors.get(this.input_) as tf.Tensor4D;
    const poolingType = this.getPoolingType();
    if (this.layout_ === MLInputOperandLayout.nchw) {
      // nchw -> nhwc
      input = tf.transpose(input, [0, 2, 3, 1]);
    }
    const windowDimensions = this.windowDimensions_;
    if (windowDimensions[0] === -1 && windowDimensions[1] === -1) {
      windowDimensions[0] = input.shape[1];
      windowDimensions[1] = input.shape[2];
    }
    let padding: 'valid'|'same'|ExplicitPadding;
    if (this.autoPad_ === MLAutoPad.explicit) {
      if (this.padding_.every(v => v === 0)) {
        padding = 'valid';
      } else {
        padding = [
          [0, 0], [this.padding_[0], this.padding_[1]],
          [this.padding_[2], this.padding_[3]], [0, 0]
        ] as ExplicitPadding;
      }
    } else {
      if (this.autoPad_ === MLAutoPad['same-upper']) {
        padding = 'same';
      } else {
        // Calculate the explicit paddings for 'same-lower'
        padding = [[0, 0], [0, 0], [0, 0], [0, 0]];
        const outputSizes = [0, 0];
        for (let i = 0; i < 2; ++i) {
          outputSizes[i] = Math.ceil(input.shape[1 + i] / this.strides_[i]);
        }
        const totalPadding: [number, number] = [0, 0];
        for (let i = 0; i < 2; ++i) {
          totalPadding[i] = this.strides_[i] * (outputSizes[i] - 1) +
              ((windowDimensions[i] - 1) * this.dilations_[i] + 1) -
              input.shape[1 + i];
        }
        for (let i = 0; i < 2; ++i) {
          padding[i + 1][0] = totalPadding[i] - Math.floor(totalPadding[i] / 2);
          padding[i + 1][1] = Math.floor(totalPadding[i] / 2);
        }
      }
    }

    let output = tf.pool(
        input, this.windowDimensions_, poolingType, padding, this.dilations_,
        this.strides_);
    if (this.layout_ === MLInputOperandLayout.nchw) {
      // nhwc -> nchw
      output = tf.transpose(output, [0, 3, 1, 2]);
    }
    return output;
  }

  abstract getPoolingType(): PoolingType;
}

export class AveragePool2d extends Pool {
  getPoolingType(): PoolingType {
    return 'avg';
  }
}

export class MaxPool2d extends Pool {
  getPoolingType(): PoolingType {
    return 'max';
  }
}
