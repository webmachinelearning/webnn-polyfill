import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../compilation';
import {PaddingMode, PadOptions} from '../model_builder';
import {Operand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Pad extends SingleOutputOperation {
  private input_: Operand;
  private padding_: Operand;
  private mode_: PaddingMode = PaddingMode.constant;
  private value_ = 0;

  constructor(input: Operand, padding: Operand, options: PadOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    utils.validateOperand(padding);
    this.padding_ = padding;
    if (options.mode !== undefined) {
      utils.assert(
          options.mode in PaddingMode, 'The mode parameter is invalid.');
      this.mode_ = options.mode;
    }
    if (options.value !== undefined) {
      this.value_ = options.value;
    }
  }

  inputs(): Operand[] {
    return [this.input_, this.padding_];
  }

  run(context: ExecutionContext): tf.Tensor {
    const input: tf.Tensor = context.getTensor(this.input_);
    const padding: tf.Tensor = context.getTensor(this.padding_);
    utils.assert(
        padding.rank === 2 && padding.dtype === 'int32' &&
            padding.shape[0] === input.rank,
        'The padding operand is invalid.');
    const paddingArray = padding.arraySync() as Array<[number, number]>;
    if (this.mode_ === PaddingMode.constant) {
      return tf.pad(input, paddingArray, this.value_);
    } else {
      if (this.mode_ === PaddingMode.edge) {
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
        return padded;
      } else {
        let mode: 'reflect'|'symmetric';
        if (this.mode_ === PaddingMode.reflection) {
          mode = 'reflect';
        } else if (this.mode_ === PaddingMode.symmetric) {
          mode = 'symmetric';
        }
        return tf.mirrorPad(input, paddingArray, mode);
      }
    }
  }
}