import * as tf from '@tensorflow/tfjs-core';

import {Binary} from './binary';

export class Sub extends Binary {
  runOp(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
    return tf.sub(a, b);
  }
}