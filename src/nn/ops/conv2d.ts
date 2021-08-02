import * as tf from '@tensorflow/tfjs-core';
import {ExplicitPadding} from '@tensorflow/tfjs-core/src/ops/conv_util';

import {MLAutoPad, MLConv2dOptions, MLFilterOperandLayout, MLInputOperandLayout} from '../graph_builder';
import {ConstantOperand, MLOperand, OutputOperand} from '../operand';
import {FusedOperation, MLOperator, SingleOutputOperation} from '../operation';
import * as utils from '../utils';

import {Clamp} from './clamp';
import {LeakyRelu} from './leaky_relu';
import {Relu, Sigmoid} from './unary';

export class Conv2d extends SingleOutputOperation implements FusedOperation {
  private input_: MLOperand;
  private filter_: MLOperand;
  private bias_: MLOperand;
  private padding_?: [number, number, number, number];
  private strides_?: [number, number];
  private dilations_?: [number, number];
  private groups_?: number;
  private inputLayout_?: MLInputOperandLayout;
  private filterLayout_?: MLFilterOperandLayout;
  private autoPad_?: MLAutoPad;
  private outputPadding_?: [number, number];
  private outputSizes_?: [number, number];
  private transpose_?: boolean;
  private activation_?: MLOperator;
  private fusedActivation_?: tf.fused.Activation;
  private leakyreluAlpha_?: number;
  private filterTensor_?: tf.Tensor4D;

  constructor(
      input: MLOperand, filter: MLOperand, options: MLConv2dOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    utils.validateOperand(filter);
    this.filter_ = filter;

    utils.assert(
        !(options.autoPad === MLAutoPad.explicit &&
          options.padding === undefined),
        'The padding parameter should be assigned when autoPad is explicit.');

    this.initOptions(
        options.padding, options.strides, options.dilations, options.groups,
        options.inputLayout, options.filterLayout, options.autoPad,
        options.transpose, options.outputPadding, options.outputSizes,
        options.bias, options.activation);
  }

  private initOptions(
      padding: [number, number, number, number] = [0, 0, 0, 0],
      strides: [number, number] = [1, 1], dilations: [number, number] = [1, 1],
      groups = 1, inputLayout: MLInputOperandLayout = MLInputOperandLayout.nchw,
      filterLayout: MLFilterOperandLayout = MLFilterOperandLayout.oihw,
      autoPad: MLAutoPad = MLAutoPad.explicit, transpose = false,
      outputPadding: [number, number] = [0, 0],
      outputSizes: [number, number] = undefined, bias: MLOperand = undefined,
      activation: MLOperator = undefined) {
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
        inputLayout in MLInputOperandLayout,
        'The input layout parameter is invalid.');
    this.inputLayout_ = inputLayout;

    utils.assert(
        filterLayout in MLFilterOperandLayout,
        'The filter layout parameter is invalid.');
    this.filterLayout_ = filterLayout;

    utils.assert(autoPad in MLAutoPad, 'The autoPad parameter is invalid.');
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

    this.bias_ = bias;
    if (this.bias_) {
      utils.validateOperand(this.bias_);
    }

    if (activation instanceof Relu) {
      this.fusedActivation_ = 'relu';
      this.activation_ = undefined;
    } else if (this.isRelu6(activation)) {
      this.fusedActivation_ = 'relu6';
      this.activation_ = undefined;
    } else if (activation instanceof LeakyRelu) {
      this.fusedActivation_ = 'leakyrelu';
      this.leakyreluAlpha_ = (activation).alpha;
      this.activation_ = undefined;
    } else if (activation instanceof Sigmoid) {
      this.fusedActivation_ = 'sigmoid';
      this.activation_ = undefined;
    } else {
      this.fusedActivation_ = undefined;
      this.activation_ = activation;
    }
  }

  isRelu6(activation: MLOperator): boolean {
    if (activation instanceof Clamp) {
      const clamp = activation;
      if (Math.abs(clamp.minScalarValue - 0.0) < 1e-5 &&
          Math.abs(clamp.maxScalarValue - 6.0) < 1e-5) {
        return true;
      }
    }
    return false;
  }

  getFusedOutputs(): OutputOperand[] {
    if (this.activation_) {
      return [this.activation_.apply(this.output)];
    } else {
      return [this.output];
    }
  }

  inputs(): MLOperand[] {
    const inputs = [this.input_, this.filter_];
    if (this.bias_) {
      inputs.push(this.bias_);
    }
    return inputs;
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    let input: tf.Tensor4D = inputTensors.get(this.input_) as tf.Tensor4D;
    let filter: tf.Tensor4D;
    let bias: tf.Tensor1D;
    let fused = false;
    if (this.bias_) {
      bias = inputTensors.get(this.bias_) as tf.Tensor1D;
    }

    // tf.conv2d input layout (nhwc): [batch, height, width, inDepth]
    if (this.inputLayout_ === MLInputOperandLayout.nchw) {
      // nchw -> nhwc
      input = tf.transpose(input, [0, 2, 3, 1]);
    }
    const inputChannels = input.shape[3];
    // tf.conv2d filter layout (hwio): [filterHeight, filterWidth, inDepth,
    // outDepth]
    if (this.filterTensor_ === undefined) {
      filter = inputTensors.get(this.filter_) as tf.Tensor4D;
      if (this.filterLayout_ === MLFilterOperandLayout.oihw) {
        filter = tf.transpose(filter, [2, 3, 1, 0]);
      } else if (this.filterLayout_ === MLFilterOperandLayout.ohwi) {
        filter = tf.transpose(filter, [1, 2, 3, 0]);
      } else if (this.filterLayout_ === MLFilterOperandLayout.ihwo) {
        filter = tf.transpose(filter, [1, 2, 0, 3]);
      }
      if (this.groups_ !== 1) {
        // filter layout hwio
        // tf.depthwiseConv2d filter layout: [filterHeight, filterWidth,
        // inChannels, channelMultiplier]
        filter = tf.transpose(filter, [0, 1, 3, 2]);
      }
      if (this.filter_ instanceof ConstantOperand) {
        this.filterTensor_ = filter;
        tf.keep(this.filterTensor_);
      }
    } else {
      filter = this.filterTensor_;
    }
    const padding: 'valid'|'same'|ExplicitPadding = utils.getPaddings(
        input, filter, this.padding_, this.strides_, this.outputPadding_,
        this.dilations_, this.autoPad_);
    let output;
    if (this.transpose_ === false) {
      if (this.groups_ === 1) {
        output = tf.fused.conv2d({
          x: input,
          filter,
          strides: this.strides_,
          pad: padding,
          dataFormat: 'NHWC',
          dilations: this.dilations_,
          bias,
          activation: this.fusedActivation_,
          leakyreluAlpha: this.leakyreluAlpha_
        });
        fused = true;
      } else if (
          this.groups_ === inputChannels && this.groups_ === filter.shape[2]) {
        if (padding === 'valid' || padding === 'same' ||
            (padding instanceof Array && padding[1][0] === padding[1][1] &&
             padding[1][0] === padding[2][0] &&
             padding[1][0] === padding[2][1])) {
          let fusedDepthwisePad: 'valid'|'same'|number;
          if (padding === 'valid' || padding === 'same') {
            fusedDepthwisePad = padding;
          } else {
            fusedDepthwisePad = padding[1][0];
          }
          output = tf.fused.depthwiseConv2d({
            x: input,
            filter,
            strides: this.strides_,
            pad: fusedDepthwisePad,
            dataFormat: 'NHWC',
            dilations: this.dilations_,
            bias,
            activation: this.fusedActivation_,
            leakyreluAlpha: this.leakyreluAlpha_
          });
          fused = true;
        } else {
          output = tf.depthwiseConv2d(
              input, filter, this.strides_, padding, 'NHWC', this.dilations_);
        }
      } else {
        throw new Error(
            'The tf.js convolution doesn\'t support groups parameter' +
            ` ${this.groups_}`);
      }
    } else {
      // transpose == true
      if (this.autoPad_ !== MLAutoPad.explicit) {
        this.outputSizes_ = [
          input.shape[1] * this.strides_[0],
          input.shape[2] * this.strides_[1],
        ];
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
    if (!fused) {
      if (bias) {
        // output is still nhwc
        output = tf.add(output, bias);
      }
      if (this.fusedActivation_ === 'relu') {
        output = tf.relu(output);
      } else if (this.fusedActivation_ === 'relu6') {
        output = tf.clipByValue(output, 0, 6);
      } else if (this.fusedActivation_ === 'leakyrelu') {
        output = tf.leakyRelu(output, this.leakyreluAlpha_);
      } else if (this.fusedActivation_ === 'sigmoid') {
        output = tf.sigmoid(output);
      } else if (this.fusedActivation_ !== undefined) {
        utils.assert(false, `The ${this.fusedActivation_} is un supported.`);
      }
    }
    if (this.inputLayout_ === MLInputOperandLayout.nchw) {
      // nhwc -> nchw
      output = tf.transpose(output, [0, 3, 1, 2]);
    }
    return output;
  }

  dispose(): void {
    if (this.filterTensor_) {
      tf.dispose(this.filterTensor_);
    }
  }
}
