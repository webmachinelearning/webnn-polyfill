import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../execution_context';
import {Operand} from '../operand_impl';
import {Operation} from '../operation';

export abstract class Unary extends Operation {
  constructor(x: Operand) {
    super([x]);
  }

  run(context: ExecutionContext): tf.Tensor {
    const x: tf.Tensor = this.getTensor(this.inputs[0], context);
    return this.runOp(x);
  }

  abstract runOp(x: tf.Tensor): tf.Tensor;
}