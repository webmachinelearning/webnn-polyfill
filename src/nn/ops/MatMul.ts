import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../execution_context';
import {Operand} from '../operand';
import {Operation} from '../operation';

export class MatMul extends Operation {
  constructor(a: Operand, b: Operand) {
    super([a, b]);
  }

  run(context: ExecutionContext): tf.Tensor {
    const a: tf.Tensor = this.getTensor(this.inputs[0], context);
    const b: tf.Tensor = this.getTensor(this.inputs[1], context);
    if (a.rank === 1 || b.rank === 1) {
      return tf.dot(a, b);
    } else {
      return tf.matMul(a, b);
    }
  }
}