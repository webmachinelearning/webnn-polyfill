import * as tf from '@tensorflow/tfjs-core';

import {MLReduceOptions} from '../graph_builder';
import {MLOperand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

abstract class Reduce extends SingleOutputOperation {
  private input_: MLOperand;
  private axes_?: number[];
  private keepDimensions_?: boolean;

  constructor(input: MLOperand, options: MLReduceOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    if (options.axes !== undefined) {
      utils.assert(
          utils.isIntegerArray(options.axes), 'The axes parameter is invalid.');
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

  inputs(): MLOperand[] {
    return [this.input_];
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    const input: tf.Tensor = inputTensors.get(this.input_);
    // accepts axis range [-r, r)
    utils.assert(
        utils.validateAxes(this.axes_, input.rank),
        `The axes must be in range [-${input.rank}, ${input.rank})`);
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

export class ReduceL1 extends Reduce {
  runOp(input: tf.Tensor, axes: number[], keepDimensions: boolean): tf.Tensor {
    return tf.sum(tf.abs(input), axes, keepDimensions);
  }
}

export class ReduceL2 extends Reduce {
  runOp(input: tf.Tensor, axes: number[], keepDimensions: boolean): tf.Tensor {
    return tf.sqrt(tf.sum(tf.pow(input, 2), axes, keepDimensions));
  }
}