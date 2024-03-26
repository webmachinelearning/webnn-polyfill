import * as tf from '@tensorflow/tfjs-core';
import {ExplicitPadding} from '@tensorflow/tfjs-core/dist/ops/conv_util';

import {MLConvTranspose2dOptions, MLConvTranspose2dFilterOperandLayout, MLInputOperandLayout} from '../graph_builder';
import {ConstantOperand, MLOperand, OutputOperand} from '../operand';
import {FusedOperation, MLActivation, SingleOutputOperation} from '../operation';
import * as utils from '../utils';

import {Clamp} from './clamp';
import {LeakyRelu} from './leaky_relu';
import {Relu, Sigmoid} from './unary';

export class ConvTranspose2d extends SingleOutputOperation
  implements FusedOperation {
  private input_: MLOperand;
  private filter_: MLOperand;
  private bias_: MLOperand;
  private padding_: [number, number, number, number];
  private strides_?: [number, number];
  private dilations_?: [number, number];
  private groups_?: number;
  private inputLayout_?: MLInputOperandLayout;
  private filterLayout_?: MLConvTranspose2dFilterOperandLayout;
  private outputPadding_?: [number, number];
  private outputSizes_?: [number, number];
  private activation_?: MLActivation;
  private fusedActivation_?: tf.fused.Activation;
  private leakyreluAlpha_?: number;
  private filterTensor_?: tf.Tensor4D;
  private needCheckOutputShape_ = true;
  private outputShape_: [number, number, number, number];

  constructor(
      input: MLOperand, filter: MLOperand,
      options: MLConvTranspose2dOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    utils.validateOperand(filter);
    this.filter_ = filter;

    this.initOptions(
        options.padding, options.strides, options.dilations, options.groups,
        options.inputLayout, options.filterLayout,
        options.outputPadding, options.outputSizes,
        options.bias, options.activation);

    this.createOutput();
  }

  private initOptions(
      padding: [number, number, number, number] = [0, 0, 0, 0],
      strides: [number, number] = [1, 1], dilations: [number, number] = [1, 1],
      groups = 1, inputLayout: MLInputOperandLayout = MLInputOperandLayout.nchw,
      filterLayout: MLConvTranspose2dFilterOperandLayout =
      MLConvTranspose2dFilterOperandLayout.iohw,
      outputPadding: [number, number] = [0, 0],
      outputSizes: [number, number] = undefined, bias: MLOperand = undefined,
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
    utils.assert(this.dilations_.every(v => v === 1),
      'The tf.conv2dTranspose does not support dilations parameter.');

    utils.assert(utils.isInteger(groups), 'The gourps parameter is invalid.');
    this.groups_ = groups;
    utils.assert(this.groups_ === 1,
      'The tf.conv2dTranspose does not support groups parameter.');

    utils.assert(
        inputLayout in MLInputOperandLayout,
        'The input layout parameter is invalid.');
    this.inputLayout_ = inputLayout;

    utils.assert(
        filterLayout in MLConvTranspose2dFilterOperandLayout,
        'The filter layout parameter is invalid.');
    this.filterLayout_ = filterLayout;

    utils.assert(
        outputSizes === undefined ||
            (utils.isIntegerArray(outputSizes) && outputSizes.length === 2),
        'The outputSizes parameter is invalid.');
    this.outputSizes_ = outputSizes;

    if (outputSizes === undefined) {
      utils.assert(
        utils.isIntegerArray(outputPadding) && outputPadding.length === 2,
        'The outputPadding parameter is invalid.');
      this.outputPadding_ = outputPadding;
    } else {
      // When the output sizes are explicitly specified, the output padding
      // values in options.outputPadding are ignored.
      this.outputPadding_ = [0, 0];
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
    const filterShape = this.filter_.shape();
    switch (this.filterLayout_) {
      case MLConvTranspose2dFilterOperandLayout.iohw:
        channels = filterShape[1];
        filterHeight = filterShape[2];
        filterWidth = filterShape[3];
        break;
      case MLConvTranspose2dFilterOperandLayout.hwoi:
        filterHeight = filterShape[0];
        filterWidth = filterShape[1];
        channels = filterShape[2];
        break;
      case MLConvTranspose2dFilterOperandLayout.ohwi:
        channels = filterShape[0];
        filterHeight = filterShape[1];
        filterWidth = filterShape[2];
        break;
      default:
        throw new Error('The filter layout is invalid.');
    }

    switch (this.inputLayout_) {
      case MLInputOperandLayout.nchw:
        [batches, , inputHeight, inputWidth] = inputShape;
        break;
      case MLInputOperandLayout.nhwc:
        [batches, inputHeight, inputWidth, ] = inputShape;
        this.outputShape_ = [batches, 0, 0, channels];
        break;
      default:
        throw new Error('The input layout is invalid.');
    }

    let outputHeight, outputWidth;
    if (this.outputSizes_ !== undefined) {
      [outputHeight, outputWidth] = this.outputSizes_;
    } else {
      // output size = (input size - 1) * stride + filter size +
      //               (filter size - 1) * (dilations - 1) -
      //               beginning padding - ending padding + output padding
      // outputSize = (inputSize - 1) * stride + (filterSize - 1) * dilation
      //              + 1 - beginningPadding - endingPadding + outputPadding
      outputHeight = (inputHeight - 1) * this.strides_[0] +
        (filterHeight - 1) * this.dilations_[0] + 1 -
        this.padding_[0] - this.padding_[1] + this.outputPadding_[0];
      outputWidth = (inputWidth - 1) * this.strides_[1] +
        (filterWidth - 1) * this.dilations_[1] + 1 -
        this.padding_[2] - this.padding_[3] + this.outputPadding_[1];
    }

    this.outputShape_ = this.inputLayout_ === MLInputOperandLayout.nchw ?
        [batches, channels, outputHeight, outputWidth] :
        [batches, outputHeight, outputWidth, channels];

    this.outputs_.push(new OutputOperand(this,
      {dataType: this.input_.dataType(), dimensions: this.outputShape_}));
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    let input: tf.Tensor4D = inputTensors.get(this.input_) as tf.Tensor4D;
    let filter: tf.Tensor4D;
    let bias: tf.Tensor1D;
    if (this.bias_) {
      bias = inputTensors.get(this.bias_) as tf.Tensor1D;
    }
    // tf.convTranspose2d input layout (nhwc): [batch, height, width, inDepth]
    if (this.inputLayout_ === MLInputOperandLayout.nchw) {
      // nchw -> nhwc
      input = tf.transpose(input, [0, 2, 3, 1]);
    }
    if (this.filterTensor_ === undefined) {
      filter = inputTensors.get(this.filter_) as tf.Tensor4D;
      // tf.conv2dTranspose filter layout (hwoi): [filterHeight, filterWidth,
      // outDepth, inDepth]
      if (this.filterLayout_ === MLConvTranspose2dFilterOperandLayout.iohw) {
        filter = tf.transpose(filter, [2, 3, 1, 0]);
      } else if (
        this.filterLayout_ === MLConvTranspose2dFilterOperandLayout.ohwi) {
        filter = tf.transpose(filter, [1, 2, 0, 3]);
      }
      if (this.groups_ !== 1) {
        // TODO
        throw new Error(
            'Unsupported the groups parameter by tfjs.convTranspose2d');
      }
      if (this.filter_ instanceof ConstantOperand) {
        this.filterTensor_ = filter;
        tf.keep(this.filterTensor_);
      }
    } else {
      filter = this.filterTensor_;
    }
    const outputShape = this.inputLayout_ === MLInputOperandLayout.nhwc ?
      this.outputShape_ : [this.outputShape_[0], this.outputShape_[2],
                           this.outputShape_[3], this.outputShape_[1]];
    const tfPadding: ExplicitPadding = utils.getPaddings(this.padding_);
    let output = tf.conv2dTranspose(
        input, filter, outputShape as [number, number, number, number],
        this.strides_, tfPadding);
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
