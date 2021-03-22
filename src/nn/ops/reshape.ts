import * as tf from '@tensorflow/tfjs-core';

import {Operand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Reshape extends SingleOutputOperation {
  private input_: Operand;
  private newShape_: number[];

  constructor(input: Operand, newShape: number[]) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    utils.assert(
        utils.isIntegerArray(newShape) && newShape.length !== 0,
        'The newShape parameter is invalid.');
    this.newShape_ = newShape;
  }

  inputs(): Operand[] {
    return [this.input_];
  }

  run(inputTensors: Map<Operand, tf.Tensor>): tf.Tensor {
    const input: tf.Tensor = inputTensors.get(this.input_);
    return tf.reshape(input, this.newShape_);
  }
}