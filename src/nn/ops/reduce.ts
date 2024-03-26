import * as tf from '@tensorflow/tfjs-core';

import {MLReduceOptions} from '../graph_builder';
import {MLOperand, OutputOperand} from '../operand';
import {SingleOutputOperation} from '../operation';
import * as utils from '../utils';

abstract class Reduce extends SingleOutputOperation {
  private input_: MLOperand;
  private axes_?: number[];
  private keepDimensions_?: boolean;
  private needCheckOutputShape_ = true;
  private outputShape_: number[];

  constructor(input: MLOperand, options: MLReduceOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    if (options.axes !== undefined) {
      utils.assert(
          utils.isUnsignedIntegerArray(options.axes),
          'The axes parameter is invalid.');
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

    this.createOutput();
  }

  inputs(): MLOperand[] {
    return [this.input_];
  }

  createOutput(): void {
    const inpAxes = this.axes_ ?? [...Array(this.input_.rank()).keys()];
    this.outputShape_ = this.input_.shape().slice();
    for (let i = 0; i < inpAxes.length; ++i) {
      this.outputShape_[inpAxes[i]] = 1;
    }
    if (!this.keepDimensions_) {
      this.outputShape_ = this.outputShape_.filter((dim) => dim !== 1);
    }
    this.outputs_.push(new OutputOperand(this,
      {dataType: this.input_.dataType(), dimensions: this.outputShape_}));
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    const input: tf.Tensor = inputTensors.get(this.input_);
    // accepts axis range [0, r)
    utils.assert(
        utils.validateAxes(this.axes_, input.rank),
        `The axes must be in range [0, ${input.rank})`);
    const output = this.runOp(input, this.axes_, this.keepDimensions_);
    if (this.needCheckOutputShape_) {
      utils.checkShape(output.shape, this.outputShape_);
      this.needCheckOutputShape_ = false;
    }
    return output;
  }

  abstract runOp(input: tf.Tensor, axes: number[], keepDimensions: boolean):
      tf.Tensor;
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

export class ReduceLogSum extends Reduce {
  runOp(input: tf.Tensor, axes: number[], keepDimensions: boolean): tf.Tensor {
    return tf.log(tf.sum(input, axes, keepDimensions));
  }
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

export class ReduceSumSquare extends Reduce {
  runOp(input: tf.Tensor, axes: number[], keepDimensions: boolean): tf.Tensor {
    return tf.sum(tf.pow(input, 2), axes, keepDimensions);
  }
}