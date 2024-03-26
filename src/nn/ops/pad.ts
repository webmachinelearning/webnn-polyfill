import * as tf from '@tensorflow/tfjs-core';

import {MLPaddingMode, MLPadOptions} from '../graph_builder';
import {MLOperand, OutputOperand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Pad extends SingleOutputOperation {
  private input_: MLOperand;
  private beginningPadding_: [number, number];
  private endingPadding_: [number, number];
  private mode_: MLPaddingMode = MLPaddingMode.constant;
  private value_ = 0;
  private needCheckOutputShape_ = true;
  private outputShape_: number[];

  constructor(
      input: MLOperand,
      beginningPadding: [number, number],
      endingPadding: [number, number],
      options: MLPadOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    utils.assert(
      utils.isUnsignedIntegerArray(beginningPadding),
      'Each element of the beginningPadding parameter should be unsigned ' +
          'interger.');
    this.beginningPadding_ = beginningPadding;
    utils.assert(
      utils.isUnsignedIntegerArray(endingPadding),
      'Each element of the endingPadding parameter should be unsigned ' +
          'interger.');
    this.endingPadding_ = endingPadding;
    if (options.mode !== undefined) {
      utils.assert(
          options.mode in MLPaddingMode, 'The mode parameter is invalid.');
      this.mode_ = options.mode;
    }
    if (options.value !== undefined) {
      this.value_ = options.value;
    }
    this.createOutput();
  }

  inputs(): MLOperand[] {
    return [this.input_];
  }

  createOutput(): void {
    this.outputShape_ = this.input_.shape().map(
      (value, index) => 
          value + this.beginningPadding_[index] + this.endingPadding_[index]);
    this.outputs_.push(new OutputOperand(this,
      {dataType: this.input_.dataType(), dimensions: this.outputShape_}));
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    const input: tf.Tensor = inputTensors.get(this.input_);
    utils.assert(
        this.beginningPadding_.length === input.shape.length,
        'The length of beginningPadding parameter should be equal to the ' +
            'lenght of input shape.');
    utils.assert(
        this.endingPadding_.length === input.shape.length,
        'The length of endingPadding parameter should be equal to the ' +
            'lenght of input shape.');
    const paddingArray: Array<[number, number]> = this.beginningPadding_.map(
        (val, index) => [val, this.endingPadding_[index]]);
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
      utils.checkShape(output.shape, this.outputShape_);
      this.needCheckOutputShape_ = false;
    }
    return output;
  }
}