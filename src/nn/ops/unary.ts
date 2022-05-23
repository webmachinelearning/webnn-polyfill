import * as tf from '@tensorflow/tfjs-core';

import {MLOperand, OutputOperand} from '../operand';
import {MLOperator, SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export abstract class Unary extends SingleOutputOperation {
  protected x_: MLOperand;
  private needCheckOutputShape_: boolean;

  constructor(x: MLOperand) {
    if (x !== undefined) {
      super(x.builder);
      utils.validateOperand(x);
      this.x_ = x;
    } else {
      super(undefined);
      this.x_ = undefined;
    }
    this.needCheckOutputShape_ = true;
  }

  inputs(): MLOperand[] {
    return [this.x_];
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    const x: tf.Tensor = inputTensors.get(this.x_);
    const output: tf.Tensor = this.runOp(x);
    if (this.needCheckOutputShape_) {
      // The output shape is the same shape as the input
      utils.checkShape(output.shape, x.shape);
      this.needCheckOutputShape_ = false;
    }
    return output;
  }

  abstract runOp(x: tf.Tensor): tf.Tensor;
}

export class Abs extends Unary {
  runOp(x: tf.Tensor): tf.Tensor {
    return tf.abs(x);
  }
}

export class Ceil extends Unary {
  runOp(x: tf.Tensor): tf.Tensor {
    return tf.ceil(x);
  }
}

export class Cos extends Unary {
  runOp(x: tf.Tensor): tf.Tensor {
    return tf.cos(x);
  }
}

export class Exp extends Unary {
  runOp(x: tf.Tensor): tf.Tensor {
    return tf.exp(x);
  }
}

export class Floor extends Unary {
  runOp(x: tf.Tensor): tf.Tensor {
    return tf.floor(x);
  }
}

export class Log extends Unary {
  runOp(x: tf.Tensor): tf.Tensor {
    return tf.log(x);
  }
}

export class Neg extends Unary {
  runOp(x: tf.Tensor): tf.Tensor {
    return tf.neg(x);
  }
}

export class Sin extends Unary {
  runOp(x: tf.Tensor): tf.Tensor {
    return tf.sin(x);
  }
}

export class Tan extends Unary {
  runOp(x: tf.Tensor): tf.Tensor {
    return tf.tan(x);
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

export class HardSwish extends UnaryMLOperator {
  runOp(x: tf.Tensor): tf.Tensor {
    return tf.div(
      tf.mul(
          x,
          tf.maximum(
              0,
              tf.minimum(
                  6,
                  tf.add(x, 3)))),
      6);
  }
}
