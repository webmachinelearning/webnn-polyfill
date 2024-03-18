import * as tf from '@tensorflow/tfjs-core';

import {MLOperand, OutputOperand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Concat extends SingleOutputOperation {
  private inputs_: MLOperand[];
  private axis_: number;
  private needCheckOutputShape_ = true;
  private outputShape_: number[];

  constructor(inputs: MLOperand[], axis: number) {
    super(inputs[0].builder);
    utils.assert(
        inputs.every(input => input instanceof MLOperand),
        'The parameter is not an operand.');
    this.inputs_ = inputs;
    utils.assert(
        utils.isUnsignedInteger(axis), 'The axis parameter is invalid.');
    this.axis_ = axis;
    this.createOutput();
  }

  inputs(): MLOperand[] {
    return this.inputs_;
  }

  createOutput(): void {
    this.outputShape_ = this.inputs_[0].shape().slice();
    for (let i = 1; i < this.inputs_.length; ++i) {
      this.outputShape_[this.axis_] += this.inputs_[i].shape()[this.axis_];
    }
    this.outputs_.push(new OutputOperand(this,
        {dataType: this.inputs_[0].dataType(), dimensions: this.outputShape_}));
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    const inputs: tf.Tensor[] = [];
    for (const input of this.inputs()) {
      inputs.push(inputTensors.get(input));
    }
    const output = tf.concat(inputs, this.axis_);
    if (this.needCheckOutputShape_) {
      utils.checkShape(output.shape, this.outputShape_);
      this.needCheckOutputShape_ = false;
    }
    return output;
  }
}
