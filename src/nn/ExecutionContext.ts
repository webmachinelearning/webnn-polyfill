import * as tf from '@tensorflow/tfjs-core';

import {Constant} from './Constant';
import {Input} from './Input';

export interface ExecutionContext {
  inputTensors: Map<Input, tf.Tensor>;
  constantTenosrs: Map<Constant, tf.Tensor>;
}