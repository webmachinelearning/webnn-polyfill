import * as tf from '@tensorflow/tfjs-core';

import {Operand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export abstract class Unary extends SingleOutputOperation {
  private x_: Operand;

  constructor(x: Operand) {
    super(x.builder);
    utils.validateOperand(x);
    this.x_ = x;
  }

  inputs(): Operand[] {
    return [this.x_];
  }

  run(inputTensors: Map<Operand, tf.Tensor>): tf.Tensor {
    const x: tf.Tensor = inputTensors.get(this.x_);
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
