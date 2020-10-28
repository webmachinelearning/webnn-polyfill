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

export class Exp extends Unary {
  runOp(x: tf.Tensor): tf.Tensor {
    return tf.exp(x);
  }
}

export class Sigmoid extends Unary {
  runOp(x: tf.Tensor): tf.Tensor {
    return tf.sigmoid(x);
  }
}

export class Sqrt extends Unary {
  runOp(x: tf.Tensor): tf.Tensor {
    return tf.sqrt(x);
  }
}

export class Tanh extends Unary {
  runOp(x: tf.Tensor): tf.Tensor {
    return tf.tanh(x);
  }
}

export class Relu extends Unary {
  runOp(x: tf.Tensor): tf.Tensor {
    return tf.relu(x);
  }
}
