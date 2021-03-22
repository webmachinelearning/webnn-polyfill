import * as tf from '@tensorflow/tfjs-core';

import {Operand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export abstract class Binary extends SingleOutputOperation {
  private a_: Operand;
  private b_: Operand;

  constructor(a: Operand, b: Operand) {
    super(a.builder);
    utils.validateOperand(a);
    this.a_ = a;
    utils.validateOperand(b);
    this.b_ = b;
  }

  inputs(): Operand[] {
    return [this.a_, this.b_];
  }

  run(inputTensors: Map<Operand, tf.Tensor>): tf.Tensor {
    const a: tf.Tensor = inputTensors.get(this.a_);
    const b: tf.Tensor = inputTensors.get(this.b_);
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

export class Pow extends Binary {
  runOp(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
    return tf.pow(a, b);
  }
}

export class MatMul extends Binary {
  runOp(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
    if (a.rank === 1 || b.rank === 1) {
      return tf.dot(a, b);
    } else {
      const rank = a.rank > b.rank ? a.rank : b.rank;
      let c = tf.matMul(a, b);
      // workaround https://github.com/tensorflow/tfjs/issues/4192
      if (c.rank !== rank) {
        c = tf.reshape(c, [1].concat(c.shape));
      }
      return c;
    }
  }
}