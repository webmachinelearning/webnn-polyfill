import * as tf from '@tensorflow/tfjs-core';

import {Binary} from './binary';

export class Mul extends Binary {
  runOp(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
    return tf.mul(a, b);
  }
}