import * as tf from '@tensorflow/tfjs-core';

import {MLClampOptions} from '../graph_builder';
import {ConstantOperand, MLOperand, OutputOperand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Clamp extends SingleOutputOperation {
  private x_: MLOperand;
  private minOperand_?: MLOperand;
  private maxOperand_?: MLOperand;
  private minScalarValue_?: number;
  private maxScalarValue_?: number;

  get minScalarValue(): number {
    return this.minScalarValue_;
  }
  get maxScalarValue(): number {
    return this.maxScalarValue_;
  }

  private getScalarValue(operand: MLOperand, minus = false): number {
    if (operand instanceof ConstantOperand) {
      const minConstant = operand;
      if (typeof minConstant.value === 'number') {
        return minConstant.value;
      }
    } else if (operand === undefined) {
      return minus ? -Infinity : +Infinity;
    }
    return undefined;
  }

  constructor(x: MLOperand, options: MLClampOptions = {}) {
    if (x !== undefined) {
      super(x.builder);
      utils.validateOperand(x);
      this.x_ = x;
    } else {
      super(undefined);
      this.x_ = undefined;
    }
    utils.validateOptionalOperand(options.minValue);
    this.minOperand_ = options.minValue;
    this.minScalarValue_ = this.getScalarValue(this.minOperand_, true);
    utils.validateOptionalOperand(options.maxValue);
    this.maxOperand_ = options.maxValue;
    this.maxScalarValue_ = this.getScalarValue(this.maxOperand_);
  }

  inputs(): MLOperand[] {
    const inputs = [this.x_];
    if (this.minOperand_) {
      inputs.push(this.minOperand_);
    }
    if (this.maxOperand_) {
      inputs.push(this.maxOperand_);
    }
    return inputs;
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    const x: tf.Tensor = inputTensors.get(this.x_);
    if (this.minOperand_) {
      if (this.maxOperand_) {
        return tf.minimum(
            tf.maximum(x, inputTensors.get(this.minOperand_)),
            inputTensors.get(this.maxOperand_));
      } else {
        return tf.maximum(x, inputTensors.get(this.minOperand_));
      }
    } else {
      if (this.maxOperand_) {
        return tf.minimum(x, inputTensors.get(this.maxOperand_));
      } else {
        return tf.clone(x);
      }
    }
  }

  apply(x: MLOperand): OutputOperand {
    this.builder_ = x.builder;
    utils.validateOperand(x);
    this.x_ = x;
    this.createOutput();
    return this.output;
  }

  runOp(x: tf.Tensor): tf.Tensor {
    utils.assert(
        this.minScalarValue_ !== undefined &&
            this.maxScalarValue_ !== undefined,
        'tf.js only supports clipByValue.');
    return tf.clipByValue(x, this.minScalarValue_, this.maxScalarValue_);
  }
}