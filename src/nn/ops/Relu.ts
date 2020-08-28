import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../ExecutionContext';
import {Operand} from '../Operand';
import {Operation} from '../Operation';

export class Relu extends Operation {
  constructor(input: Operand) {
    super([input]);
  }

  run(context: ExecutionContext): tf.Tensor {
    const input: tf.Tensor = this.getTensor(this.inputs[0], context);
    return tf.relu(input);
  }
}