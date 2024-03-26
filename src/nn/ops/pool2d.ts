import * as tf from '@tensorflow/tfjs-core';
import {ExplicitPadding} from '@tensorflow/tfjs-core/dist/ops/conv_util';

import {MLInputOperandLayout, MLPooling2dOptions, MLRoundingType} from '../graph_builder';
import {MLOperand, OutputOperand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

type PoolingType = 'avg'|'l2'|'max';

export abstract class Pool extends SingleOutputOperation {
  protected input_: MLOperand;
  protected windowDimensions_?: [number, number];
  protected padding_?: [number, number, number, number];
  protected strides_?: [number, number];
  protected dilations_?: [number, number];
  protected groups_?: number;
  protected layout_?: MLInputOperandLayout;
  protected roundingType_?: MLRoundingType;
  protected outputSizes_?: [number, number];
  private needCheckOutputShape_ = true;
  private outputShape_: [number, number, number, number];

  constructor(input: MLOperand, options: MLPooling2dOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;

    this.initOptions(
        options.windowDimensions, options.padding, options.strides,
        options.dilations, options.layout, options.roundingType,
        options.outputSizes);
    this.createOutput();
  }

  private initOptions(
      windowDimensions: [number, number],
      padding: [number, number, number, number] = [0, 0, 0, 0],
      strides: [number, number] = [1, 1], dilations: [number, number] = [1, 1],
      layout: MLInputOperandLayout = MLInputOperandLayout.nchw,
      roundingType: MLRoundingType = MLRoundingType.floor,
      outputSizes: [number, number] = undefined) {
    utils.assert(
      layout in MLInputOperandLayout, 'The layout parameter is invalid.');
    this.layout_ = layout;

    if (windowDimensions) {
      utils.assert(
        utils.isUnsignedIntegerArray(windowDimensions) &&
        windowDimensions.length === 2,
        'The windowDimensions parameter is invalid.');
      this.windowDimensions_ = windowDimensions;
    } else {
      this.windowDimensions_ = layout === MLInputOperandLayout.nchw ?
          this.input_.shape().slice(2) as [number, number] :
          this.input_.shape().slice(1, 3) as [number, number];
    }

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
        layout in MLInputOperandLayout, 'The layout parameter is invalid.');
    this.layout_ = layout;

    utils.assert(
        roundingType in MLRoundingType,
        'The roundingType parameter is invalid.');
    this.roundingType_ = roundingType;

    if (outputSizes) {
      utils.assert(
          utils.isIntegerArray(outputSizes) && outputSizes.length === 2,
          'The outputSizes parameter is invalid.');
    }

    this.outputSizes_ = outputSizes;
  }

  inputs(): MLOperand[] {
    return [this.input_];
  }

  createOutput(): void {
    let outputHeight;
    let outputWidth;
    if (this.outputSizes_ !== undefined) {
      outputHeight = this.outputSizes_[0];
      outputWidth = this.outputSizes_[1];
    } else {
      const dimRoundingMode =
          this.roundingType_ === MLRoundingType.floor ? 'floor' : 'ceil';
      [outputHeight, outputWidth] = this.calculateOutputSizes(
        this.input_.shape() as [number, number, number, number],
        dimRoundingMode);
    }

    const inputShape = this.input_.shape();
    switch (this.layout_) {
      case MLInputOperandLayout.nchw:
        this.outputShape_ =
            [inputShape[0], inputShape[1], outputHeight, outputWidth];
        break;
      case MLInputOperandLayout.nhwc:
        this.outputShape_ =
            [inputShape[0], outputHeight, outputWidth, inputShape[3]];
        break;
      default:
        throw new Error('The layout is invalid.');
    }
  
    this.outputs_.push(new OutputOperand(this,
      {dataType: this.input_.dataType(), dimensions: this.outputShape_}));
  }
  /**
   * Calcuate output sizes for a given input shape and round type.
   * @param inputShape - A shape array
   * @param roundingType - String value: 'ceil' | 'floor' | 'round'
   * @returns output sizes array of [outputHeight, outputWidth].
   */
  calculateOutputSizes(
      inputShape: [number, number, number, number],
      roundingType?: 'ceil' | 'floor' | 'round'): [number, number] {
    // nhwc layout
    let inputHeight = inputShape[1];
    let inputWidth = inputShape[2];
    if (this.layout_ === MLInputOperandLayout.nchw) {
      inputHeight = inputShape[2];
      inputWidth = inputShape[3];
    }
    const windowHeight = this.windowDimensions_[0];
    const windowWidth = this.windowDimensions_[1];
    let roundFun;
  
    if (roundingType === undefined) {
      roundFun = Math.trunc;
    } else {
      switch (roundingType) {
        case 'ceil':
          roundFun = Math.ceil;
          break;
        case 'floor':
          roundFun = Math.floor;
          break;
        case 'round':
          roundFun = Math.round;
          break;
        default:
          throw new Error('The rounding type is invalid.');
      }
    }

    const effectiveWindowHeight =
        windowHeight + (windowHeight - 1) * (this.dilations_[0] - 1);
    const effectiveWindowWidth =
        windowWidth + (windowWidth - 1) * (this.dilations_[1] - 1);
    const outputHeight = 1 + roundFun((inputHeight - effectiveWindowHeight +
        this.padding_[0] + this.padding_[1]) / this.strides_[0]);
    const outputWidth = 1 + roundFun((inputWidth - effectiveWindowWidth +
        this.padding_[2] + this.padding_[3]) / this.strides_[1]);

    return [outputHeight, outputWidth];
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    let input: tf.Tensor4D = inputTensors.get(this.input_) as tf.Tensor4D;
    if (this.layout_ === MLInputOperandLayout.nchw) {
      // nchw -> nhwc
      input = tf.transpose(input, [0, 2, 3, 1]);
    }

    const windowDimensions = this.windowDimensions_;
    if (windowDimensions[0] === -1 && windowDimensions[1] === -1) {
      windowDimensions[0] = input.shape[1];
      windowDimensions[1] = input.shape[2];
    }

    let dimRoundingMode: 'ceil'|'floor'|'round';
    if (this.outputSizes_ !== undefined) {
      let isValidOutputSizes = false;
      for (const t of [undefined, 'ceil', 'floor', 'round']) {
        const [outputRows, outputCols] = this.calculateOutputSizes(input.shape,
            t as 'ceil'|'floor'|'round');
        if (this.outputSizes_[0] === outputRows
            && this.outputSizes_[1] === outputCols) {
          dimRoundingMode = t as 'ceil'|'floor'|'round';
          isValidOutputSizes = true;
          break;
        }
      }
      utils.assert(
          isValidOutputSizes,
          `The outputSizes [${this.outputSizes_}] is invalid.`);
    } else {
      dimRoundingMode =
          this.roundingType_ === MLRoundingType.floor ? 'floor' : 'ceil';
    }

    const poolingType = this.getPoolingType();
    let padding: 'valid'|'same'|ExplicitPadding;

    if (this.padding_.every(v => v === 0)) {
      padding = 'valid';
      // unset dimRoundingMode when using valid pad, refer to
      //   https://github.com/tensorflow/tfjs/blob/master/tfjs-core/src/ops/conv_util.ts#L604
      dimRoundingMode = undefined;
    } else {
      padding = [
        [0, 0], [this.padding_[0], this.padding_[1]],
        [this.padding_[2], this.padding_[3]], [0, 0]
      ] as ExplicitPadding;
    }

    let output;
    if (poolingType === 'l2') {
      input = tf.pow(input, 2);
      output = tf.sqrt(
        tf.pool(input, this.windowDimensions_, 'avg', padding, this.dilations_,
        this.strides_, dimRoundingMode));
    } else {
      output = tf.pool(
        input, this.windowDimensions_, poolingType, padding, this.dilations_,
        this.strides_, dimRoundingMode);
    }

    if (this.layout_ === MLInputOperandLayout.nchw) {
      // nhwc -> nchw
      output = tf.transpose(output, [0, 3, 1, 2]);
    }

    if (this.needCheckOutputShape_) {
      // check output shape by TF.js with own calculated output shape
      utils.checkShape(output.shape, this.outputShape_);
      this.needCheckOutputShape_ = false;
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

export class L2Pool2d extends Pool {
  getPoolingType(): PoolingType {
    return 'l2';
  }
}
