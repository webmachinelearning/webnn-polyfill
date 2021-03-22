import * as tf from '@tensorflow/tfjs-core';
import {ExplicitPadding} from '@tensorflow/tfjs-core/src/ops/conv_util';

import {AutoPad, Conv2dOptions, FilterOperandLayout, InputOperandLayout} from '../model_builder';
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
  private autoPad_?: AutoPad;
  private outputPadding_?: [number, number];
  private outputSizes_?: [number, number];
  private transpose_?: boolean;

  constructor(input: Operand, filter: Operand, options: Conv2dOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    utils.validateOperand(filter);
    this.filter_ = filter;
    this.initOptions(
        options.padding, options.strides, options.dilations, options.groups,
        options.inputLayout, options.filterLayout, options.autoPad,
        options.transpose, options.outputPadding, options.outputSizes);
  }

  private initOptions(
      padding: [number, number, number, number] = [0, 0, 0, 0],
      strides: [number, number] = [1, 1], dilations: [number, number] = [1, 1],
      groups = 1, inputLayout: InputOperandLayout = InputOperandLayout.nchw,
      filterLayout: FilterOperandLayout = FilterOperandLayout.oihw,
      autoPad: AutoPad = AutoPad.explicit, transpose = false,
      outputPadding: [number, number] = [0, 0],
      outputSizes: [number, number] = undefined) {
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

    utils.assert(
        inputLayout in InputOperandLayout,
        'The input layout parameter is invalid.');
    this.inputLayout_ = inputLayout;

    utils.assert(
        filterLayout in FilterOperandLayout,
        'The filter layout parameter is invalid.');
    this.filterLayout_ = filterLayout;

    utils.assert(autoPad in AutoPad, 'The autoPad parameter is invalid.');
    this.autoPad_ = autoPad;

    this.transpose_ = transpose;

    if (this.transpose_) {
      utils.assert(
          utils.isIntegerArray(outputPadding) && outputPadding.length === 2,
          'The outputPadding parameter is invalid.');
      this.outputPadding_ = outputPadding;

      utils.assert(
          outputSizes === undefined ||
              (utils.isIntegerArray(outputSizes) && outputSizes.length === 2),
          'The outputSizes parameter is invalid.');
      this.outputSizes_ = outputSizes;
    } else {
      this.outputPadding_ = [0, 0];
      this.outputSizes_ = undefined;
    }
  }

  inputs(): Operand[] {
    return [this.input_, this.filter_];
  }

  run(inputTensors: Map<Operand, tf.Tensor>): tf.Tensor {
    let input: tf.Tensor4D = inputTensors.get(this.input_) as tf.Tensor4D;
    let filter: tf.Tensor4D = inputTensors.get(this.filter_) as tf.Tensor4D;

    // tf.conv2d input layout (nhwc): [batch, height, width, inDepth]
    if (this.inputLayout_ === InputOperandLayout.nchw) {
      // nchw -> nhwc
      input = tf.transpose(input, [0, 2, 3, 1]);
    }
    const inputChannels = input.shape[3];
    // tf.conv2d filter layout (hwio): [filterHeight, filterWidth, inDepth,
    // outDepth]
    if (this.filterLayout_ === FilterOperandLayout.oihw) {
      filter = tf.transpose(filter, [2, 3, 1, 0]);
    } else if (this.filterLayout_ === FilterOperandLayout.ohwi) {
      filter = tf.transpose(filter, [1, 2, 3, 0]);
    }

    let output;
    if (this.transpose_ === false) {
      if (this.groups_ === 1) {
        // WebNN padding:
        //   [beginning_height, ending_height, beginning_width, ending_width]
        // tf.conv2d NHWC should be in the following form:
        //   [[0, 0], [pad_top,pad_bottom], [pad_left, pad_right], [0, 0]]
        let padding: 'valid'|'same'|ExplicitPadding;
        if (this.autoPad_ === AutoPad.explicit) {
          if (this.padding_.every(v => v === 0)) {
            padding = 'valid';
          } else {
            padding = [
              [0, 0], [this.padding_[0], this.padding_[1]],
              [this.padding_[2], this.padding_[3]], [0, 0]
            ] as ExplicitPadding;
          }
        } else {
          // Calculate the explicit paddings for 'same-upper'
          if (this.autoPad_ === AutoPad['same-upper']) {
            padding = [[0, 0], [0, 0], [0, 0], [0, 0]];
            const outputSizes = [0, 0];
            for (let i = 0; i < 2; ++i) {
              outputSizes[i] = Math.ceil(input.shape[1 + i] / this.strides_[i]);
            }
            const totalPadding: [number, number] = [0, 0];
            for (let i = 0; i < 2; ++i) {
              totalPadding[i] = this.strides_[i] * (outputSizes[i] - 1) +
                  this.outputPadding_[i] +
                  ((filter.shape[i] - 1) * this.dilations_[i] + 1) -
                  input.shape[1 + i];
            }
            for (let i = 0; i < 2; ++i) {
              padding[i + 1][0] =
                  totalPadding[i] - Math.floor(totalPadding[i] / 2);
              padding[i + 1][1] = Math.floor(totalPadding[i] / 2);
            }
          } else {
            // 'same-lower'
            padding = 'same';
          }
        }
        output = tf.conv2d(
            input, filter, this.strides_, padding, 'NHWC', this.dilations_);
      } else if (
          this.groups_ === inputChannels && this.groups_ === filter.shape[3]) {
        let padding: 'valid'|'same'|number;
        if (this.autoPad_ === AutoPad.explicit) {
          utils.assert(
              this.padding_.every(v => v === this.padding_[0]),
              'tf.depthwiseConv2d only supports the same padding value.');
          padding = this.padding_[0] === 0 ? 'valid' : this.padding_[0];
        } else {
          if (this.autoPad_ === AutoPad['same-lower']) {
            padding = 'same';
          } else {
            throw new Error(
                'tf.depthwiseConv2d only supports the same-lower auto pad.');
          }
        }
        // filter layout hwio
        // tf.depthwiseConv2d filter layout: [filterHeight, filterWidth,
        // inChannels, channelMultiplier]
        filter = tf.transpose(filter, [0, 1, 3, 2]);
        output = tf.depthwiseConv2d(
            input, filter, this.strides_, padding, 'NHWC', this.dilations_);
      } else {
        throw new Error(
            'The tf.js convolution doesn\'t support groups parameter' +
            ` ${this.groups_}`);
      }
    } else {
      // transpose == true
      utils.assert(
          this.dilations_.every(v => v === 1),
          'tf.conv2dTranspose only supports dilation 1.');
      let padding: 'valid'|'same'|number;
      if (this.autoPad_ === AutoPad.explicit) {
        utils.assert(
            this.padding_.every(v => v === this.padding_[0]),
            'tf.conv2dTranspose only supports the same padding value.');
        padding = this.padding_[0] === 0 ? 'valid' : this.padding_[0];
      } else {
        if (this.autoPad_ === AutoPad['same-lower']) {
          padding = 'same';
          this.outputSizes_ = [
            input.shape[1] * this.strides_[0],
            input.shape[2] * this.strides_[1],
          ];
        } else {
          throw new Error(
              'tf.conv2dTranspose only supports the same-lower auto pad.');
        }
      }
      // tf.conv2dTranspose outputShape: [batch, height, width, outDepth]
      const outputShape: [number, number, number, number] =
          [input.shape[0], 0, 0, filter.shape[2]];
      if (this.outputSizes_ === undefined) {
        for (let i = 0; i < 2; ++i) {
          outputShape[i + 1] = this.strides_[i] * (input.shape[i + 1] - 1) +
              this.outputPadding_[i] +
              ((filter.shape[i] - 1) * this.dilations_[i] + 1) -
              this.padding_[i * 2] - this.padding_[i * 2 + 1];
        }
      } else {
        outputShape[1] = this.outputSizes_[0];
        outputShape[2] = this.outputSizes_[1];
      }
      output = tf.conv2dTranspose(
          input, filter, outputShape, this.strides_, padding);
    }
    if (this.inputLayout_ === InputOperandLayout.nchw) {
      // nhwc -> nchw
      output = tf.transpose(output, [0, 3, 1, 2]);
    }
    return output;
  }
}
