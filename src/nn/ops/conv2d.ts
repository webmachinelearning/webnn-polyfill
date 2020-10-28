import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../execution_context';
import {Conv2dOptions} from '../conv2d_options';
import {Operand} from '../operand_impl';
import {OperandLayout} from '../operand_layout';
import {Operation} from '../operation';
import * as utils from '../utils';

export class Conv2d extends Operation {
  private padding_: [number, number, number, number] = [0, 0, 0, 0];
  private strides_: [number, number] = [1, 1];
  private dilations_: [number, number] = [1, 1];
  private groups_ = 1;
  private layout_: OperandLayout = OperandLayout.nchw;

  constructor(input: Operand, filter: Operand, options: Conv2dOptions = {}) {
    super([input, filter]);

    if (options.padding !== undefined) {
      utils.assert(
          utils.isIntegerArray(options.padding) && options.padding.length === 4,
          'The options.padding parameter is invalid.');
      this.padding_ = options.padding;
    }

    if (options.strides !== undefined) {
      utils.assert(
          utils.isIntegerArray(options.strides) && options.strides.length === 2,
          'The options.strides parameter is invalid.');
      this.strides_ = options.strides;
    }

    if (options.dilations !== undefined) {
      utils.assert(
          utils.isIntegerArray(options.dilations) &&
              options.dilations.length === 2,
          'The options.dilations parameter is invalid.');
      this.dilations_ = options.dilations;
    }

    if (options.groups !== undefined) {
      utils.assert(
          utils.isInteger(options.groups),
          'The options.gourps parameter is invalid.');
      this.groups_ = options.groups;
    }

    if (options.layout !== undefined) {
      utils.assert(
          options.layout in OperandLayout,
          'The options.layout parameter is invalid.');
      this.layout_ = options.layout;
    }
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