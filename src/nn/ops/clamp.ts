import * as tf from '@tensorflow/tfjs-core';

import {MLClampOptions} from '../graph_builder';
import {ConstantOperand, MLOperand, OutputOperand} from '../operand';
import {SingleOutputOperation, MLOperator} from '../operation';
import * as utils from '../utils';

export class Clamp extends SingleOutputOperation implements MLOperator {
  private x_: MLOperand;
  private minOperand_?: MLOperand;
  private maxOperand_?: MLOperand;
  private minScalarValue_?: number;
  private maxScalarValue_?: number;

  get minScalarValue(): number {return this.minScalarValue_;}
  get maxScalarValue(): number {return this.maxScalarValue_;}

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
    if (this.minOperand_ instanceof ConstantOperand) {
      const minConstant = this.minOperand_;
      if (typeof minConstant.value === 'number') {
        this.minScalarValue_ = minConstant.value;
      }
    } else if (this.minOperand_ === undefined) {
      this.minScalarValue_ = -Infinity;
    } else {
      this.minScalarValue_ = undefined;
    }
    utils.validateOptionalOperand(options.maxValue);
    this.maxOperand_ = options.maxValue;
    if (this.maxOperand_ instanceof ConstantOperand) {
      const maxConstant = this.maxOperand_;
      if (typeof maxConstant.value === 'number') {
        this.maxScalarValue_ = maxConstant.value;
      }
    } else if (this.maxOperand_ === undefined) {
      this.maxScalarValue_ = +Infinity;
    } else {
      this.maxScalarValue_ = undefined;
    }
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
    utils.assert(this.minScalarValue_ !== undefined &&
        this.maxScalarValue_ !== undefined, 'tf.js only supports clipByValue.');
    return tf.clipByValue(x, this.minScalarValue_, this.maxScalarValue_);
  }
}