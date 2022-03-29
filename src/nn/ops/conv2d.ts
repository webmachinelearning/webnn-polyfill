import * as tf from '@tensorflow/tfjs-core';
import {ExplicitPadding} from '@tensorflow/tfjs-core/dist/ops/conv_util';

import {MLAutoPad, MLConv2dOptions, MLConv2dFilterOperandLayout, MLInputOperandLayout} from '../graph_builder';
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
  private filterLayout_?: MLConv2dFilterOperandLayout;
  private autoPad_?: MLAutoPad;
  private activation_?: MLOperator;
  private fusedActivation_?: tf.fused.Activation;
  private leakyreluAlpha_?: number;
  private filterTensor_?: tf.Tensor4D;
  private needCheckOutputShape_ = true;

  constructor(
      input: MLOperand, filter: MLOperand, options: MLConv2dOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    utils.validateOperand(filter);
    this.filter_ = filter;

    this.initOptions(
        options.padding, options.strides, options.dilations, options.groups,
        options.inputLayout, options.filterLayout, options.autoPad, 
        options.bias, options.activation);
  }

  private initOptions(
      padding: [number, number, number, number] = [0, 0, 0, 0],
      strides: [number, number] = [1, 1], dilations: [number, number] = [1, 1],
      groups = 1, inputLayout: MLInputOperandLayout = MLInputOperandLayout.nchw,
      filterLayout: 
      MLConv2dFilterOperandLayout = MLConv2dFilterOperandLayout.oihw,
      autoPad: MLAutoPad = MLAutoPad.explicit, bias: MLOperand = undefined,
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
        filterLayout in MLConv2dFilterOperandLayout,
        'The filter layout parameter is invalid.');
    this.filterLayout_ = filterLayout;

    utils.assert(autoPad in MLAutoPad, 'The autoPad parameter is invalid.');
    this.autoPad_ = autoPad;
    
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
      if (Math.abs(clamp.minValue - 0.0) < 1e-5 &&
          Math.abs(clamp.maxValue - 6.0) < 1e-5) {
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
    if (this.filterTensor_ === undefined) {
      filter = inputTensors.get(this.filter_) as tf.Tensor4D;
  
    // tf.conv2d filter layout (hwio): [filterHeight, filterWidth, inDepth,
    // outDepth]
    if (this.filterLayout_ === MLConv2dFilterOperandLayout.oihw) {
      filter = tf.transpose(filter, [2, 3, 1, 0]);
    } else if (this.filterLayout_ === MLConv2dFilterOperandLayout.ohwi) {
      filter = tf.transpose(filter, [1, 2, 3, 0]);
    } else if (this.filterLayout_ === MLConv2dFilterOperandLayout.ihwo) {
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
    const padding: ExplicitPadding = utils.getPaddings(input, filter,
        this.padding_, this.strides_, this.dilations_, this.autoPad_);
    let output;
    
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
      if ((padding instanceof Array && padding[1][0] === padding[1][1] &&
            padding[1][0] === padding[2][0] &&
            padding[1][0] === padding[2][1])) {
        const fusedDepthwisePad: number = padding[1][0];
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
    if (this.needCheckOutputShape_) {
      const effectiveFilterHeight =
          filter.shape[0] + (filter.shape[0] - 1) * (this.dilations_[0] - 1);
      const effectiveFilterWidth =
          filter.shape[1] + (filter.shape[1] - 1) * (this.dilations_[1] - 1);
      // output size = 1 +
      //     (input size - filter size - (filter size - 1) * (dilation - 1) +
      //      beginning padding + ending padding) / stride
      const outputHeight =
          1 + Math.floor((input.shape[1] - effectiveFilterHeight +
              padding[1][0] + padding[1][1]) / this.strides_[0]);
      const outputWidth =
          1 + Math.floor((input.shape[2] - effectiveFilterWidth +
              padding[2][0] + padding[2][1]) / this.strides_[1]);
      // A depthwise conv2d operation is a variant of grouped convolution, used
      // in models like the MobileNet, where the
      //   options.groups = input_channels = output_channels
      const outputChannels =
          this.groups_ !== 1 ? filter.shape[2] : filter.shape[3];
      const outputShape = new Array(4);
      outputShape[0] = input.shape[0];
      outputShape[1] = outputHeight;
      outputShape[2] = outputWidth;
      outputShape[3] = outputChannels;
      if (this.inputLayout_ === MLInputOperandLayout.nchw) {
        // nhwc -> nchw
        outputShape[1] = outputChannels;
        outputShape[2] = outputHeight;
        outputShape[3] = outputWidth;
      }
      utils.checkShape(output.shape, outputShape);
      this.needCheckOutputShape_ = false;
    }
    return output;
  }

  dispose(): void {
    if (this.filterTensor_) {
      tf.dispose(this.filterTensor_);
    }
  }
}