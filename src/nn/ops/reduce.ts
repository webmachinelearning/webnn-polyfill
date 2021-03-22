import * as tf from '@tensorflow/tfjs-core';

import {ReduceOptions} from '../model_builder';
import {Operand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

abstract class Reduce extends SingleOutputOperation {
  private input_: Operand;
  private axes_?: number[];
  private keepDimensions_?: boolean;

  constructor(input: Operand, options: ReduceOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    if (options.axes !== undefined) {
      utils.assert(
          utils.isIntegerArray(options.axes) &&
              options.axes.every(v => v > 0 || v === -1),
          'The starts parameter is invalid.');
      this.axes_ = options.axes;
    } else {
      this.axes_ = undefined;
    }
    if (options.keepDimensions !== undefined) {
      utils.assert(
          utils.isBoolean(options.keepDimensions),
          'The keepDimensions parameter is not a boolean.');
      this.keepDimensions_ = options.keepDimensions;
    } else {
      this.keepDimensions_ = false;
    }
  }

  inputs(): Operand[] {
    return [this.input_];
  }

  run(inputTensors: Map<Operand, tf.Tensor>): tf.Tensor {
    const input: tf.Tensor = inputTensors.get(this.input_);
    // tf.mean accepts axis range [-r, r)
    return this.runOp(input, this.axes_, this.keepDimensions_);
  }

  abstract runOp(input: tf.Tensor, axes: number[], keepDimensions: boolean):
      tf.Tensor;
}

export class ReduceLogSumExp extends Reduce {
  runOp(input: tf.Tensor, axes: number[], keepDimensions: boolean): tf.Tensor {
    return tf.logSumExp(input, axes, keepDimensions);
  }
}

export class ReduceMax extends Reduce {
  runOp(input: tf.Tensor, axes: number[], keepDimensions: boolean): tf.Tensor {
    return tf.max(input, axes, keepDimensions);
  }
}

export class ReduceMean extends Reduce {
  runOp(input: tf.Tensor, axes: number[], keepDimensions: boolean): tf.Tensor {
    return tf.mean(input, axes, keepDimensions);
  }
}

export class ReduceMin extends Reduce {
  runOp(input: tf.Tensor, axes: number[], keepDimensions: boolean): tf.Tensor {
    return tf.min(input, axes, keepDimensions);
  }
}

export class ReduceProduct extends Reduce {
  runOp(input: tf.Tensor, axes: number[], keepDimensions: boolean): tf.Tensor {
    return tf.prod(input, axes, keepDimensions);
  }
}

export class ReduceSum extends Reduce {
  runOp(input: tf.Tensor, axes: number[], keepDimensions: boolean): tf.Tensor {
    return tf.sum(input, axes, keepDimensions);
  }
}
