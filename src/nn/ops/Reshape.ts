import { Operation } from "../Operation";
import { Operand } from "../Operand";
import { ExecutionContext } from "../ExecutionContext";
import * as utils from '../utils';

import * as tf from '@tensorflow/tfjs-core';

export class Reshape extends Operation {
  private newShape_: number[];

  constructor(input: Operand, newShape: number[]) {
    super([input]);
    utils.assert(utils.isNumberArray(newShape), 'The newShape parameter is invalid.');
    this.newShape_ = newShape;
  }

  run(context: ExecutionContext): tf.Tensor {
    const input: tf.Tensor = this.getTensor(this.inputs[0], context);
    return tf.reshape(input, this.newShape_);
  }
}