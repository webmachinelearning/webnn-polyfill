import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../compilation';
import {ClampOptions} from '../model_builder';
import {Operand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Clamp extends SingleOutputOperation {
  private x_: Operand;
  private minValue_?: Operand;
  private maxValue_?: Operand;

  constructor(x: Operand, options: ClampOptions = {}) {
    super(x.builder);
    utils.validateOperand(x);
    this.x_ = x;
    utils.validateOptionalOperand(options.minValue);
    this.minValue_ = options.minValue;
    utils.validateOptionalOperand(options.maxValue);
    this.maxValue_ = options.maxValue;
  }

  inputs(): Operand[] {
    const inputs = [this.x_];
    if (this.minValue_) {
      inputs.push(this.minValue_);
    }
    if (this.maxValue_) {
      inputs.push(this.maxValue_);
    }
    return inputs;
  }

  run(context: ExecutionContext): tf.Tensor {
    const x: tf.Tensor = context.getTensor(this.x_);
    if (this.minValue_) {
      if (this.maxValue_) {
        return tf.minimum(
            tf.maximum(x, context.getTensor(this.minValue_)),
            context.getTensor(this.maxValue_));
      } else {
        return tf.maximum(x, context.getTensor(this.minValue_));
      }
    } else {
      if (this.maxValue_) {
        return tf.minimum(x, context.getTensor(this.maxValue_));
      } else {
        return x;
      }
    }
  }
}