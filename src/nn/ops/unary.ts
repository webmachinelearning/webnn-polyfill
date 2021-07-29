import * as tf from '@tensorflow/tfjs-core';

import {MLOperand, OutputOperand} from '../operand';
import {MLOperator, SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export abstract class Unary extends SingleOutputOperation {
  protected x_: MLOperand;

  constructor(x: MLOperand) {
    if (x !== undefined) {
      super(x.builder);
      utils.validateOperand(x);
      this.x_ = x;
    } else {
      super(undefined);
      this.x_ = undefined;
    }
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

export class Sqrt extends Unary {
  runOp(x: tf.Tensor): tf.Tensor {
    return tf.sqrt(x);
  }
}

export abstract class UnaryMLOperator extends Unary implements MLOperator {
  apply(x: MLOperand): OutputOperand {
    this.builder_ = x.builder;
    utils.validateOperand(x);
    this.x_ = x;
    this.createOutput();
    return this.output;
  }
}

export class Sigmoid extends UnaryMLOperator {
  runOp(x: tf.Tensor): tf.Tensor {
    return tf.sigmoid(x);
  }
}

export class Tanh extends UnaryMLOperator {
  runOp(x: tf.Tensor): tf.Tensor {
    return tf.tanh(x);
  }
}

export class Relu extends UnaryMLOperator {
  runOp(x: tf.Tensor): tf.Tensor {
    return tf.relu(x);
  }
}
