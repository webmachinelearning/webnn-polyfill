import * as tf from '@tensorflow/tfjs-core';

import {MLPaddingMode, MLPadOptions} from '../graph_builder';
import {MLOperand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Pad extends SingleOutputOperation {
  private input_: MLOperand;
  private padding_: MLOperand;
  private mode_: MLPaddingMode = MLPaddingMode.constant;
  private value_ = 0;
  private needCheckOutputShape_ = true;

  constructor(
      input: MLOperand, padding: MLOperand, options: MLPadOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    utils.validateOperand(padding);
    this.padding_ = padding;
    if (options.mode !== undefined) {
      utils.assert(
          options.mode in MLPaddingMode, 'The mode parameter is invalid.');
      this.mode_ = options.mode;
    }
    if (options.value !== undefined) {
      this.value_ = options.value;
    }
  }

  inputs(): MLOperand[] {
    return [this.input_, this.padding_];
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    const input: tf.Tensor = inputTensors.get(this.input_);
    const padding: tf.Tensor = inputTensors.get(this.padding_);
    utils.assert(
        padding.rank === 2 && padding.dtype === 'int32' &&
            padding.shape[0] === input.rank,
        'The padding operand is invalid.');
    const paddingArray = padding.arraySync() as Array<[number, number]>;
    const outputShape = input.shape.map(
        (val, index) => val + paddingArray[index][0] + paddingArray[index][1]);
    let output;
    if (this.mode_ === MLPaddingMode.constant) {
      output = tf.pad(input, paddingArray, this.value_);
    } else {
      if (this.mode_ === MLPaddingMode.edge) {
        const edgePaddings: Array<[number, number]> =
            new Array(paddingArray.length);
        let padded: tf.Tensor = input;
        for (;;) {
          for (let i = 0; i < paddingArray.length; ++i) {
            edgePaddings[i] = [0, 0];
            for (let j = 0; j < 2; ++j) {
              if (paddingArray[i][j] > 0) {
                edgePaddings[i][j] = 1;
                paddingArray[i][j] -= 1;
              } else {
                edgePaddings[i][j] = 0;
              }
            }
          }
          if (edgePaddings.every(value => value[0] === 0 && value[1] === 0)) {
            break;
          }
          padded = tf.mirrorPad(padded, edgePaddings, 'symmetric');
        }
        output = padded;
      } else {
        let mode: 'reflect'|'symmetric';
        if (this.mode_ === MLPaddingMode.reflection) {
          mode = 'reflect';
        } else if (this.mode_ === MLPaddingMode.symmetric) {
          mode = 'symmetric';
        }
        output = tf.mirrorPad(input, paddingArray, mode);
      }
    }
    if (this.needCheckOutputShape_) {
      utils.checkShape(output.shape, outputShape);
      this.needCheckOutputShape_ = false;
    }
    return output;
  }
}