import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from './compilation';
import {ConstantOperand, InputOperand, Operand, OutputOperand} from './operand';
import * as utils from './utils';

export abstract class Operation {
  inputs: Operand[] = [];
  outputs: OutputOperand[] = [];

  constructor(inputs: Operand[]) {
    utils.assert(
        inputs.every(input => input instanceof Operand),
        'The inputs parameter is invalid.');
    this.inputs = inputs;
    this.outputs.push(new OutputOperand(this, this.inputs[0].builder));
  }

  get output(): OutputOperand {
    return this.outputs[0];
  }

  protected getTensor(operand: Operand, context: ExecutionContext): tf.Tensor {
    if (operand instanceof ConstantOperand) {
      return context.constantTenosrs.get(operand);
    } else if (operand instanceof InputOperand) {
      return context.inputTensors.get(operand);
    } else if (operand instanceof OutputOperand) {
      return operand.operation.run(context);
    } else {
      throw new Error('The operand is invalid.');
    }
  }

  abstract run(context: ExecutionContext): tf.Tensor;
}