import * as tf from '@tensorflow/tfjs-core';

import {MLReduceOptions} from '../graph_builder';
import {MLOperand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

abstract class Reduce extends SingleOutputOperation {
  private input_: MLOperand;
  private axes_?: number[];
  private keepDimensions_?: boolean;
  private needCheckOutputShape_ = true;

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
    const output = this.runOp(input, this.axes_, this.keepDimensions_);
    if (this.needCheckOutputShape_) {
      const inpAxes = this.axes_ ?? [...Array(input.rank).keys()];
      let outputShape = input.shape.slice();
      for (let i = 0; i < inpAxes.length; ++i) {
        if (inpAxes[i] < 0) {
          inpAxes[i] = input.rank + inpAxes[i];
        }
        outputShape[inpAxes[i]] = 1;
      }
      if (!this.keepDimensions_) {
        outputShape = outputShape.filter((dim, axis) =>
          !(dim === 1 && inpAxes.indexOf(axis) !== -1));
      }
      utils.checkShape(output.shape, outputShape);
      this.needCheckOutputShape_ = false;
    }
    return output;
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