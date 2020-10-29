import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../compilation';
import {Conv2dOptions, OperandLayout} from '../model_builder';
import {Operand} from '../operand';
import {Operation} from '../operation';
import * as utils from '../utils';

export class Conv2d extends Operation {
  private padding_: [number, number, number, number];
  private strides_: [number, number];
  private dilations_: [number, number];
  private groups_: number;
  private layout_: OperandLayout;

  constructor(input: Operand, filter: Operand, options: Conv2dOptions = {}) {
    super([input, filter]);
    this.initOptions(
        options.padding, options.strides, options.dilations, options.groups,
        options.layout);
  }

  private initOptions(
      padding: [number, number, number, number] = [0, 0, 0, 0],
      strides: [number, number] = [1, 1], dilations: [number, number] = [1, 1],
      groups = 1, layout: OperandLayout = OperandLayout.nchw) {
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

    utils.assert(utils.isInteger(groups), 'The gourps parameter is invalid.');
    this.groups_ = groups;

    utils.assert(layout in OperandLayout, 'The layout parameter is invalid.');
    this.layout_ = layout;
  }

  run(context: ExecutionContext): tf.Tensor {
    let input: tf.Tensor4D =
        this.getTensor(this.inputs[0], context) as tf.Tensor4D;
    let filter: tf.Tensor4D =
        this.getTensor(this.inputs[1], context) as tf.Tensor4D;
    utils.assert(
        this.padding_.every(v => v === this.padding_[0]),
        'The tf.conv2d only supports the same padding value.');
    const padding = this.padding_[0];
    let inputChannels: number;
    if (this.layout_ === OperandLayout.nchw) {
      // nchw -> nhwc
      input = input.transpose([0, 2, 3, 1]);
      inputChannels = input.shape[1];
      // nchw filter: [output_channels, input_channels/groups, height, width]
      // nhwc filter: [height, width, input_channels/groups, output_channels]
      filter = filter.transpose([2, 3, 1, 0]);
    } else {
      // 'NHWC'
      inputChannels = input.shape[3];
    }
    let output;
    if (this.groups_ === 1) {
      // tf.conv2d filter: [filterHeight, filterWidth, inDepth, outDepth].
      output = tf.conv2d(
          input, filter, this.strides_, padding, 'NHWC', this.dilations_);
    } else if (
        this.groups_ === inputChannels && this.groups_ === filter.shape[3]) {
      // webnn filter: [height, width, input_channels/groups, output_channels]
      // tf.depthwiseConv2d filter: [filterHeight, filterWidth, inChannels,
      // channelMultiplier].
      filter = filter.transpose([0, 1, 3, 2]);
      output = tf.depthwiseConv2d(
          input, filter, this.strides_, padding, 'NHWC', this.dilations_);
    } else {
      throw new Error(
          'The tf.js convolution doesn\'t support groups parameter' +
          ` ${this.groups_}`);
    }
    if (this.layout_ === OperandLayout.nchw) {
      // nhwc -> nchw
      output = output.transpose([0, 3, 1, 2]);
    }
    return output;
  }
}