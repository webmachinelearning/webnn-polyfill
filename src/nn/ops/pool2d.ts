import * as tf from '@tensorflow/tfjs-core';
import {ExplicitPadding} from '@tensorflow/tfjs-core/dist/ops/conv_util';

import {MLAutoPad, MLInputOperandLayout, MLPooling2dOptions, MLRoundingType} from '../graph_builder';
import {MLOperand} from '../operand';
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
  private autoPad_?: MLAutoPad;
  protected roundingType_?: MLRoundingType;
  protected outputSizes_?: [number, number];
  private needCheckOutputShape_ = true;

  constructor(input: MLOperand, options: MLPooling2dOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    this.initOptions(
        options.windowDimensions, options.padding, options.strides,
        options.dilations, options.layout, options.autoPad,
        options.roundingType, options.outputSizes);
  }

  private initOptions(
      windowDimensions: [number, number] = [-1, -1],
      padding: [number, number, number, number] = [0, 0, 0, 0],
      strides: [number, number] = [1, 1], dilations: [number, number] = [1, 1],
      layout: MLInputOperandLayout = MLInputOperandLayout.nchw,
      autoPad: MLAutoPad = MLAutoPad.explicit,
      roundingType: MLRoundingType = MLRoundingType.floor,
      outputSizes: [number, number] = undefined) {
    utils.assert(
        utils.isIntegerArray(windowDimensions) && windowDimensions.length === 2,
        'The padding parameter is invalid.');
    this.windowDimensions_ = windowDimensions;

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

    utils.assert(autoPad in MLAutoPad, 'The autoPad parameter is invalid.');
    this.autoPad_ = autoPad;

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

  /**
   * Calcuate output sizes for a given input shape and round type.
   * @param inputShape - A shape array of [N, H, W, C]
   * @param roundingType - String value: 'ceil' | 'floor' | 'round'
   * @returns output sizes array of [outputHeight, outputWidth].
   */
  calculateOutputSizes(
      inputShape:[number, number, number, number],
      roundingType?: 'ceil' | 'floor' | 'round'): [number, number] {
    // nhwc layout
    const inputHeight = inputShape[1];
    const inputWidth = inputShape[2];
    const windowHeight = this.windowDimensions_[0];
    const windowWidth = this.windowDimensions_[1];
    let paddingBeginningHeight = this.padding_[0];
    let paddingEndingHeight = this.padding_[1];
    let paddingBeginningWidth = this.padding_[2];
    let paddingEndingWidth = this.padding_[3];

    if (this.autoPad_ !== MLAutoPad.explicit) {
      [paddingBeginningHeight, paddingEndingHeight] =
          utils.computeImplicitPaddingForAutoPad(
              this.autoPad_, this.dilations_[0], inputHeight, windowHeight,
              this.strides_[0], paddingBeginningHeight, paddingEndingHeight);
      [paddingBeginningWidth, paddingEndingWidth] =
          utils.computeImplicitPaddingForAutoPad(
              this.autoPad_, this.dilations_[1], inputWidth, windowWidth,
              this.strides_[1], paddingBeginningWidth, paddingEndingWidth);
    }

    let roundFun;
    if (roundingType === undefined) {
      roundFun = Math.trunc;
    } else {
      switch(roundingType) {
        case 'ceil': {
          roundFun = Math.ceil;
          break;
        }
        case 'floor': {
          roundFun = Math.floor;
          break;
        }
        case 'round': {
          roundFun = Math.round;
          break;
        }
        default: {
          break;
        }
      }
    }

    const effectiveWindowHeight =
        windowHeight + (windowHeight - 1) * (this.dilations_[0] - 1);
    const effectiveWindowWidth =
        windowWidth + (windowWidth - 1) * (this.dilations_[1] - 1);
    const outputHeight = 1 + roundFun((inputHeight - effectiveWindowHeight +
        paddingBeginningHeight + paddingEndingHeight) / this.strides_[0]);
    const outputWidth = 1 + roundFun((inputWidth - effectiveWindowWidth +
        paddingBeginningWidth + paddingEndingWidth) / this.strides_[1]);

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
    if (this.autoPad_ === MLAutoPad.explicit) {
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
    } else {
      if (this.autoPad_ === MLAutoPad['same-upper']) {
        padding = 'same';
        // unset dimRoundingMode when using same pad, refer to
        //   https://github.com/tensorflow/tfjs/blob/master/tfjs-core/src/ops/conv_util.ts#L604
        dimRoundingMode = undefined;
      } else {
        // Calculate the explicit paddings for 'same-lower'
        padding = [[0, 0], [0, 0], [0, 0], [0, 0]];
        const outputSizes = [0, 0];
        for (let i = 0; i < 2; ++i) {
          outputSizes[i] = Math.ceil(input.shape[1 + i] / this.strides_[i]);
        }
        const totalPadding: [number, number] = [0, 0];
        for (let i = 0; i < 2; ++i) {
          totalPadding[i] = this.strides_[i] * (outputSizes[i] - 1) +
              ((windowDimensions[i] - 1) * this.dilations_[i] + 1) -
              input.shape[1 + i];
        }
        for (let i = 0; i < 2; ++i) {
          padding[i + 1][0] = totalPadding[i] - Math.floor(totalPadding[i] / 2);
          padding[i + 1][1] = Math.floor(totalPadding[i] / 2);
        }
      }
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
      let outputHeight;
      let outputWidth;
      let outputShape;
      if (this.outputSizes_ !== undefined) {
        outputHeight = this.outputSizes_[0];
        outputWidth = this.outputSizes_[1];
      } else {
        dimRoundingMode =
            this.roundingType_ === MLRoundingType.floor ? 'floor' : 'ceil';
        [outputHeight, outputWidth] =
            this.calculateOutputSizes(input.shape, dimRoundingMode);
      }
      if (this.layout_ === MLInputOperandLayout.nchw) {
        outputShape =
            [input.shape[0], input.shape[3], outputHeight, outputWidth];
      } else {
        outputShape =
            [input.shape[0], outputHeight, outputWidth, input.shape[3]];
      }

      // check output shape by TF.js with own calculated output shape
      utils.checkShape(output.shape, outputShape);
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
