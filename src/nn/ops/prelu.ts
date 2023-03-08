import * as tf from '@tensorflow/tfjs-core';
import {Binary} from './binary';

export class PRelu extends Binary {
  runOp(x: tf.Tensor, slope: tf.Tensor): tf.Tensor {
    return tf.prelu(x, slope);
  }
}