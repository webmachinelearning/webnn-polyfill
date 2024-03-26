import * as tf from '@tensorflow/tfjs-core';

import {MLOperand, OutputOperand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export abstract class Binary extends SingleOutputOperation {
  private a_: MLOperand;
  private b_: MLOperand;
  private outputShape_: number[];
  private needCheckOutputShape_ = true;

  constructor(a: MLOperand, b: MLOperand) {
    super(a.builder);
    utils.validateOperand(a);
    this.a_ = a;
    utils.validateOperand(b);
    this.b_ = b;
    this.createOutput();
  }

  createOutput(): void {
    if (this instanceof MatMul) {
      const rankA = this.a_.rank();
      utils.assert(rankA >= 2, 'The inputA is at least 2-D.');
      const rankB = this.b_.rank();
      utils.assert(rankB >= 2, 'The inputB is at least 2-D.');
      this.outputShape_ = utils.getBroadcastShape(
        this.a_.shape().slice(0, -2),
        this.b_.shape().slice(0, -2));
      this.outputShape_.push(this.a_.shape()[rankA - 2]);
      this.outputShape_.push(this.b_.shape()[rankB - 1]);
    } else {
      this.outputShape_ = utils.getBroadcastShape(
          this.a_.shape(), this.b_.shape());
    }
    this.outputs_.push(new OutputOperand(this,
        {dataType: this.a_.dataType(), dimensions: this.outputShape_}));
  }

  inputs(): MLOperand[] {
    return [this.a_, this.b_];
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    const a: tf.Tensor = inputTensors.get(this.a_);
    const b: tf.Tensor = inputTensors.get(this.b_);
    const output = this.runOp(a, b);
    if (this.needCheckOutputShape_) {
      utils.checkShape(output.shape, this.outputShape_);
      this.needCheckOutputShape_ = false;
    }
    return output;
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
    const rank = a.rank > b.rank ? a.rank : b.rank;
    let c = tf.matMul(a, b);
    // workaround https://github.com/tensorflow/tfjs/issues/4192
    if (c.rank !== rank) {
      c = tf.reshape(c, [1].concat(c.shape));
    }
    return c;
  }
}