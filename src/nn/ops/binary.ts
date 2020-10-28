import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../execution_context';
import {Operand} from '../operand_impl';
import {Operation} from '../operation';

export abstract class Binary extends Operation {
  constructor(a: Operand, b: Operand) {
    super([a, b]);
  }

  run(context: ExecutionContext): tf.Tensor {
    const a: tf.Tensor = this.getTensor(this.inputs[0], context);
    const b: tf.Tensor = this.getTensor(this.inputs[1], context);
    return this.runOp(a, b);
  }

  abstract runOp(a: tf.Tensor, b: tf.Tensor): tf.Tensor;
}

export class Add extends Binary {
  runOp(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
    return tf.add(a, b);
  }
}

export class Sub extends Binary {
  runOp(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
    return tf.sub(a, b);
  }
}

export class Mul extends Binary {
  runOp(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
    return tf.mul(a, b);
  }
}

export class Div extends Binary {
  runOp(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
    return tf.div(a, b);
  }
}

export class Max extends Binary {
  runOp(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
    return tf.maximum(a, b);
  }
}

export class Min extends Binary {
  runOp(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
    return tf.minimum(a, b);
  }
}
