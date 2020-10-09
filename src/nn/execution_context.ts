import * as tf from '@tensorflow/tfjs-core';

import {ConstantOperand} from './constant_operand';
import {InputOperand} from './input_operand';

export interface ExecutionContext {
  inputTensors: Map<InputOperand, tf.Tensor>;
  constantTenosrs: Map<ConstantOperand, tf.Tensor>;
}