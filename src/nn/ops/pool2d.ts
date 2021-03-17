import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../compilation';
import {AutoPad, InputOperandLayout, Pooling2dOptions} from '../model_builder';
import {Operand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

type PoolingType = 'avg'|'max';

export abstract class Pool extends SingleOutputOperation {
  protected input_: Operand;
  protected windowDimensions_?: [number, number];
  protected padding_?: [number, number, number, number];
  protected strides_?: [number, number];
  protected dilations_?: [number, number];
  protected groups_?: number;
  protected layout_?: InputOperandLayout;
  private autoPad_?: AutoPad;

  constructor(input: Operand, options: Pooling2dOptions = {}) {
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
      layout: InputOperandLayout = InputOperandLayout.nchw,
      autoPad: AutoPad = AutoPad.explicit) {
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
        layout in InputOperandLayout, 'The layout parameter is invalid.');
    this.layout_ = layout;

    utils.assert(autoPad in AutoPad, 'The autoPad parameter is invalid.');
    this.autoPad_ = autoPad;
  }

  inputs(): Operand[] {
    return [this.input_];
  }

  run(context: ExecutionContext): tf.Tensor {
    let input: tf.Tensor4D = context.getTensor(this.input_) as tf.Tensor4D;
    let padding: 'valid'|'same'|number;
    if (this.autoPad_ === AutoPad.explicit) {
      utils.assert(
          this.padding_.every(v => v === this.padding_[0]),
          'tf.pool only supports the same padding value.');
      padding = this.padding_[0] === 0 ? 'valid' : this.padding_[0];
    } else {
      if (this.autoPad_ === AutoPad['same-lower']) {
        padding = 'same';
      } else {
        throw new Error('tf.pool only supports the same-lower auto pad.');
      }
    }
    const poolingType = this.getPoolingType();
    if (this.layout_ === InputOperandLayout.nchw) {
      // nchw -> nhwc
      input = tf.transpose(input, [0, 2, 3, 1]);
    }
    const windowDimensions = this.windowDimensions_;
    if (windowDimensions[0] === -1 && windowDimensions[1] === -1) {
      windowDimensions[0] = input.shape[1];
      windowDimensions[1] = input.shape[2];
    }
    let output = tf.pool(
        input, this.windowDimensions_, poolingType, padding, this.dilations_,
        this.strides_);
    if (this.layout_ === InputOperandLayout.nchw) {
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
