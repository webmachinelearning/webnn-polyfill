import * as tf from '@tensorflow/tfjs-core';

import {MLClampOptions} from '../graph_builder';
import {MLOperand, OutputOperand} from '../operand';
import {SingleOutputOperation, MLOperator} from '../operation';
import * as utils from '../utils';

export class Clamp extends SingleOutputOperation implements MLOperator {
  private x_: MLOperand;
  private minValue_?: MLOperand;
  private maxValue_?: MLOperand;

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
    this.minValue_ = options.minValue;
    utils.validateOptionalOperand(options.maxValue);
    this.maxValue_ = options.maxValue;
  }

  inputs(): MLOperand[] {
    const inputs = [this.x_];
    if (this.minValue_) {
      inputs.push(this.minValue_);
    }
    if (this.maxValue_) {
      inputs.push(this.maxValue_);
    }
    return inputs;
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    const x: tf.Tensor = inputTensors.get(this.x_);
    if (this.minValue_) {
      if (this.maxValue_) {
        return tf.minimum(
            tf.maximum(x, inputTensors.get(this.minValue_)),
            inputTensors.get(this.maxValue_));
      } else {
        return tf.maximum(x, inputTensors.get(this.minValue_));
      }
    } else {
      if (this.maxValue_) {
        return tf.minimum(x, inputTensors.get(this.maxValue_));
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
}