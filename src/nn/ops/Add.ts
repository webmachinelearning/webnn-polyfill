import * as tf from '@tensorflow/tfjs-core';

import {Binary} from './Binary';

export class Add extends Binary {
  runOp(a: tf.Tensor, b: tf.Tensor): tf.Tensor {
    return tf.add(a, b);
  }
}