import * as tf from '@tensorflow/tfjs-core';
import {ExplicitPadding} from '@tensorflow/tfjs-core/src/ops/conv_util';

import {ExecutionContext} from '../compilation';
import {Conv2dOptions, InputOperandLayout, FilterOperandLayout} from '../model_builder';
import {Operand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Conv2d extends SingleOutputOperation {
  private input_: Operand;
  private filter_: Operand;
  private padding_?: [number, number, number, number];
  private strides_?: [number, number];
  private dilations_?: [number, number];
  private groups_?: number;
  private inputLayout_?: InputOperandLayout;
  private filterLayout_?: FilterOperandLayout;

  constructor(input: Operand, filter: Operand, options: Conv2dOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    utils.validateOperand(filter);
    this.filter_ = filter;
    this.initOptions(
        options.padding, options.strides, options.dilations, options.groups,
        options.inputLayout, options.filterLayout);
  }

  private initOptions(
      padding: [number, number, number, number] = [0, 0, 0, 0],
      strides: [number, number] = [1, 1], dilations: [number, number] = [1, 1],
      groups = 1, inputLayout: InputOperandLayout = InputOperandLayout.nchw,
      filterLayout: FilterOperandLayout = FilterOperandLayout.oihw) {
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

    utils.assert(inputLayout in InputOperandLayout,
        'The input layout parameter is invalid.');
    this.inputLayout_ = inputLayout;

    utils.assert(filterLayout in FilterOperandLayout,
        'The filter layout parameter is invalid.');
    this.filterLayout_ = filterLayout;
  }

  inputs(): Operand[] {
    return [this.input_, this.filter_];
  }

  run(context: ExecutionContext): tf.Tensor {
    let input: tf.Tensor4D = context.getTensor(this.input_) as tf.Tensor4D;
    let filter: tf.Tensor4D = context.getTensor(this.filter_) as tf.Tensor4D;
    let inputChannels: number;
    if (this.inputLayout_ === InputOperandLayout.nchw) {
      // nchw -> nhwc
      inputChannels = input.shape[1];
      input = input.transpose([0, 2, 3, 1]);
    } else {
      // 'NHWC'
      inputChannels = input.shape[3];
    }
    // tf.conv2d filter layout: [filterHeight, filterWidth, inDepth, outDepth]
    if (this.filterLayout_ === FilterOperandLayout.oihw) {
      filter = filter.transpose([2, 3, 1, 0]);
    } else if (this.filterLayout_ === FilterOperandLayout.ohwi) {
      filter = filter.transpose([1, 2, 3, 0]);
    }
    let output;
    if (this.groups_ === 1) {
      let padding: 'valid'|'same'|number|ExplicitPadding;
      if (this.padding_.every(v => v === 0)) {
        padding = 'valid';
      } else {
        // WebNN padding:
        //   [beginning_height, ending_height, beginning_width, ending_width]
        // tf.conv2d NHWC should be in the following form:
        //   [[0, 0], [pad_top,pad_bottom], [pad_left, pad_right], [0, 0]]
        padding = [
          [0, 0], [this.padding_[0], this.padding_[1]],
          [this.padding_[2], this.padding_[3]], [0, 0]
        ] as ExplicitPadding;
      }
      output = tf.conv2d(
          input, filter, this.strides_, padding, 'NHWC', this.dilations_);
    } else if (
        this.groups_ === inputChannels && this.groups_ === filter.shape[3]) {
      utils.assert(
          this.padding_.every(v => v === this.padding_[0]),
          'The tf.depthwiseConv2d only supports the same padding value.');
      const padding = this.padding_[0];
      // webnn filter: [height, width, input_channels/groups, output_channels]
      // tf.depthwiseConv2d filter: [filterHeight, filterWidth, inChannels,
      // channelMultiplier].
      filter = filter.transpose([0, 1, 3, 2]);
      output = tf.depthwiseConv2d(
          input, filter, this.strides_, padding, 'NHWC', this.dilations_,
          'floor');
    } else {
      throw new Error(
          'The tf.js convolution doesn\'t support groups parameter' +
          ` ${this.groups_}`);
    }
    if (this.inputLayout_ === InputOperandLayout.nchw) {
      // nhwc -> nchw
      output = output.transpose([0, 3, 1, 2]);
    }
    return output;
  }
}
