import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../execution_context';
import {Operand} from '../operand';
import {OperandLayout} from '../operand_layout';
import {Operation} from '../operation';
import * as utils from '../utils';

export abstract class Pool extends Operation {
  protected windowDimensions_: [number, number];
  protected padding_: [number, number, number, number];
  protected strides_: [number, number];
  protected dilations_: [number, number];
  protected groups_: number;
  protected layout_: OperandLayout;

  constructor(
      input: Operand, windowDimensions: [number, number] = [-1, -1],
      padding: [number, number, number, number] = [0, 0, 0, 0],
      strides: [number, number] = [1, 1], dilations: [number, number] = [1, 1],
      layout: OperandLayout = OperandLayout.nchw) {
    super([input]);

    utils.assert(
        utils.isNumberArray(windowDimensions) && windowDimensions.length === 2,
        'The padding parameter is invalid.');
    this.windowDimensions_ = windowDimensions;

    utils.assert(
        utils.isNumberArray(padding) && padding.length === 4,
        'The padding parameter is invalid.');
    this.padding_ = padding;

    utils.assert(
        utils.isNumberArray(strides) && strides.length === 2,
        'The strides parameter is invalid.');
    this.strides_ = strides;

    utils.assert(
        utils.isNumberArray(dilations) && dilations.length === 2,
        'The dilations parameter is invalid.');
    this.dilations_ = dilations;

    utils.assert(layout in OperandLayout, 'The layout parameter is invalid.');
    this.layout_ = layout;
  }

  run(context: ExecutionContext): tf.Tensor {
    let input: tf.Tensor4D =
        this.getTensor(this.inputs[0], context) as tf.Tensor4D;
    utils.assert(
        this.padding_.every(v => v === this.padding_[0]),
        'The tf.conv2d only supports the same padding value.');
    const padding = this.padding_[0];
    const poolingType = this.getPoolingType();
    if (this.layout_ === OperandLayout.nchw) {
      // nchw -> nhwc
      input = input.transpose([0, 2, 3, 1]);
    }
    const windowDimensions = this.windowDimensions_;
    if (windowDimensions[0] === -1 && windowDimensions[1] === -1) {
      windowDimensions[0] = input.shape[1];
      windowDimensions[1] = input.shape[2];
    }
    let output = tf.pool(
        input, this.windowDimensions_, poolingType, padding, this.dilations_,
        this.strides_);
    if (this.layout_ === OperandLayout.nchw) {
      // nhwc -> nchw
      output = output.transpose([0, 3, 1, 2]);
    }
    return output;
  }

  abstract getPoolingType(): 'avg'|'max';
}