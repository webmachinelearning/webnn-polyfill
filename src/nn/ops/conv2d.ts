import * as tf from '@tensorflow/tfjs-core';
import {ExplicitPadding} from '@tensorflow/tfjs-core/dist/ops/conv_util';

import {MLConv2dOptions, MLConv2dFilterOperandLayout, MLInputOperandLayout} from '../graph_builder';
import {ConstantOperand, MLOperand, OutputOperand} from '../operand';
import {FusedOperation, MLActivation, SingleOutputOperation} from '../operation';
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
  private activation_?: MLActivation;
  private fusedActivation_?: tf.fused.Activation;
  private leakyreluAlpha_?: number;
  private filterTensor_?: tf.Tensor4D;
  private needCheckOutputShape_ = true;
  private outputShape_: [number, number, number, number];

  constructor(
      input: MLOperand, filter: MLOperand, options: MLConv2dOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    utils.validateOperand(filter);
    this.filter_ = filter;

    this.initOptions(
        options.padding, options.strides, options.dilations, options.groups,
        options.inputLayout, options.filterLayout,
        options.bias, options.activation);

    this.createOutput();
  }

  private initOptions(
      padding: [number, number, number, number] = [0, 0, 0, 0],
      strides: [number, number] = [1, 1], dilations: [number, number] = [1, 1],
      groups = 1, inputLayout: MLInputOperandLayout = MLInputOperandLayout.nchw,
      filterLayout: 
      MLConv2dFilterOperandLayout = MLConv2dFilterOperandLayout.oihw,
      bias: MLOperand = undefined,
      activation: MLActivation = undefined) {
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

  isRelu6(activation: MLActivation): boolean {
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

  createOutput(): void {
    let batches, channels, inputHeight, inputWidth, filterHeight, filterWidth;
    const inputShape = this.input_.shape();
    switch (this.inputLayout_) {
      case MLInputOperandLayout.nchw:
        [batches, , inputHeight, inputWidth] = inputShape;
        break;
      case MLInputOperandLayout.nhwc:
        [batches, inputHeight, inputWidth, ] = inputShape;
        break;
      default:
        throw new Error('The input layout is invalid.');
    }

    const filterShape = this.filter_.shape();
    switch (this.filterLayout_) {
      case MLConv2dFilterOperandLayout.oihw:
        channels = filterShape[0];
        filterHeight = filterShape[2];
        filterWidth = filterShape[3];
        break;
      case MLConv2dFilterOperandLayout.hwio:
        channels = filterShape[3];
        filterHeight = filterShape[0];
        filterWidth = filterShape[1];
        break;
      case MLConv2dFilterOperandLayout.ihwo:
        channels = filterShape[3];
        filterHeight = filterShape[1];
        filterWidth = filterShape[2];
        break;
      case MLConv2dFilterOperandLayout.ohwi:
        channels = filterShape[0];
        filterHeight = filterShape[1];
        filterWidth = filterShape[2];
        break;
      default:
        throw new Error('The filter layout is invalid.');
    }

    const effectiveFilterHeight =
        filterHeight + (filterHeight - 1) * (this.dilations_[0] - 1);
    const effectiveFilterWidth =
        filterWidth + (filterWidth- 1) * (this.dilations_[1] - 1);

    // output size = 1 +
    //     (input size - filter size - (filter size - 1) * (dilation - 1) +
    //      beginning padding + ending padding) / stride
    const outputHeight =
        1 + Math.floor((inputHeight - effectiveFilterHeight +
          this.padding_[0] + this.padding_[1]) / this.strides_[0]);
    const outputWidth =
        1 + Math.floor((inputWidth - effectiveFilterWidth +
          this.padding_[2] + this.padding_[3]) / this.strides_[1]);

    this.outputShape_ = [batches, outputHeight, outputWidth, channels];
    if (this.inputLayout_ === MLInputOperandLayout.nchw) {
      // nhwc -> nchw
      this.outputShape_ = [batches, channels, outputHeight, outputWidth];
    }

    this.outputs_.push(new OutputOperand(this,
      {dataType: this.input_.dataType(), dimensions: this.outputShape_}));
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

    const tfPadding: ExplicitPadding = utils.getPaddings(this.padding_);
    let output;
    if (this.groups_ === 1) {
      output = tf.fused.conv2d({
        x: input,
        filter,
        strides: this.strides_,
        pad: tfPadding,
        dataFormat: 'NHWC',
        dilations: this.dilations_,
        bias,
        activation: this.fusedActivation_,
        leakyreluAlpha: this.leakyreluAlpha_
      });
      fused = true;
    } else if (
        this.groups_ === inputChannels && this.groups_ === filter.shape[2]) {
      if ((this.padding_ instanceof Array &&
        this.padding_[0] === this.padding_[1] &&
        this.padding_[0] === this.padding_[2] &&
        this.padding_[0] === this.padding_[3])) {
        const fusedDepthwisePad: number = this.padding_[0];
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
            input, filter, this.strides_, tfPadding, 'NHWC',
            this.dilations_);
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
      utils.checkShape(output.shape, this.outputShape_);
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