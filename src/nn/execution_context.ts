import * as tf from '@tensorflow/tfjs-core';

import {Constant} from './constant';
import {Input} from './input';

export interface ExecutionContext {
  inputTensors: Map<Input, tf.Tensor>;
  constantTenosrs: Map<Constant, tf.Tensor>;
}