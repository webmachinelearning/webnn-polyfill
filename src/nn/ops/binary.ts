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
      const rankB = this.b_.rank();
      if (rankA === 1 && rankB === 1) {
        this.outputShape_ = []; // scalar
      } else if (rankA >= 2 && rankB === 1) {
        this.outputShape_ = this.a_.shape().slice();
        this.outputShape_[rankA - 1] = 1;
      } else if (rankA === 1 && rankB >= 2) {
        this.outputShape_ = this.b_.shape().slice();
        this.outputShape_[rankB - 2] = 1;
      } else if (rankA >= 2 && rankB >= 2) {
        this.outputShape_ = utils.getBroadcastShape(
          this.a_.shape().slice(0, -2),
          this.b_.shape().slice(0, -2));
        this.outputShape_.push(this.a_.shape()[rankA - 2]);
        this.outputShape_.push(this.b_.shape()[rankB - 1]);
      }
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
    if (a.rank === 1) {
      if (b.rank === 1) {
        return tf.dot(a, b);
      } else {
        // a is 1-D, convert to a 2-D tensor by prepending a 1 to its dimesions
        return tf.matMul(tf.reshape(a, [1, -1]), b);
      }
    } else {
      if (b.rank === 1) {
        // b is 1-D, convert to a 2-D tensor by appending a 1 to its dimesions
        return tf.matMul(a, tf.reshape(b, [-1, 1]));
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
}