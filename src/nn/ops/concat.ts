import * as tf from '@tensorflow/tfjs-core';

import {Operand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Concat extends SingleOutputOperation {
  private inputs_: Operand[];
  private axis_: number;

  constructor(inputs: Operand[], axis: number) {
    utils.assert(
        inputs.every(input => input instanceof Operand),
        'The parameter is not an operand.');
    super(inputs[0].builder);
    this.inputs_ = inputs;
    utils.assert(utils.isInteger(axis), 'The axis parameter is invalid.');
    this.axis_ = axis;
  }

  inputs(): Operand[] {
    return this.inputs_;
  }

  run(inputTensors: Map<Operand, tf.Tensor>): tf.Tensor {
    const inputs: tf.Tensor[] = [];
    for (const input of this.inputs()) {
      inputs.push(inputTensors.get(input));
    }
    return tf.concat(inputs, this.axis_);
  }
}
