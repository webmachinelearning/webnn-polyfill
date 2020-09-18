import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../execution_context';
import {Operand} from '../operand';
import {Operation} from '../operation';

export class Relu extends Operation {
  constructor(input: Operand) {
    super([input]);
  }

  run(context: ExecutionContext): tf.Tensor {
    const input: tf.Tensor = this.getTensor(this.inputs[0], context);
    return tf.relu(input);
  }
}