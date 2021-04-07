import * as tf from '@tensorflow/tfjs-core';

import {MLOperand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Concat extends SingleOutputOperation {
  private inputs_: MLOperand[];
  private axis_: number;

  constructor(inputs: MLOperand[], axis: number) {
    utils.assert(
        inputs.every(input => input instanceof MLOperand),
        'The parameter is not an operand.');
    super(inputs[0].builder);
    this.inputs_ = inputs;
    utils.assert(utils.isInteger(axis), 'The axis parameter is invalid.');
    this.axis_ = axis;
  }

  inputs(): MLOperand[] {
    return this.inputs_;
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    const inputs: tf.Tensor[] = [];
    for (const input of this.inputs()) {
      inputs.push(inputTensors.get(input));
    }
    return tf.concat(inputs, this.axis_);
  }
}
