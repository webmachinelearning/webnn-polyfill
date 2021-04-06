import * as tf from '@tensorflow/tfjs-core';

import {MLOperand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export abstract class Unary extends SingleOutputOperation {
  private x_: MLOperand;

  constructor(x: MLOperand) {
    super(x.builder);
    utils.validateOperand(x);
    this.x_ = x;
  }

  inputs(): MLOperand[] {
    return [this.x_];
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
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
