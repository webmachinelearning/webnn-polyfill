import * as tf from '@tensorflow/tfjs-core';

import {Binary} from './Binary';

export class Mul extends Binary {
  runOp(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
    return tf.mul(a, b);
  }
}