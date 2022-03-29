import * as tf from '@tensorflow/tfjs-core';

import {MLSplitOptions} from '../graph_builder';
import {MLOperand, OutputOperand} from '../operand';
import {Operation} from '../operation';
import * as utils from '../utils';

export class Split extends Operation {
  private input_: MLOperand;
  private splits_: number|number[];
  private axis_?: number;
  private needCheckOutputShape_ = true;

  constructor(
      input: MLOperand, splits: number|number[], options: MLSplitOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    utils.assert(
        utils.isInteger(splits) || utils.isIntegerArray(splits as number[]),
        'The splits parameter is invalid.');
    this.splits_ = splits;
    utils.assert(
        options.axis === undefined || utils.isInteger(options.axis),
        'The options.axis is invalid.');
    this.axis_ = options.axis ?? 0;

    // Prepare outputs.
    const numOutputs =
        utils.isInteger(splits) ? splits : (splits as number[]).length;
    for (let i = 0; i < numOutputs; ++i) {
      this.outputs.push(new OutputOperand(this));
    }
  }

  inputs(): MLOperand[] {
    return [this.input_];
  }

  computeImpl(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor[] {
    const input: tf.Tensor = inputTensors.get(this.input_);
    const outputs = tf.split(input, this.splits_, this.axis_);
    if (this.needCheckOutputShape_) {
      const axis = this.axis_ >= 0 ? this.axis_ : this.axis_ + input.rank;
      let sliceSizes = [];
      if (typeof this.splits_ === 'number') {
        sliceSizes =
            new Array(this.splits_).fill(input.shape[axis] / this.splits_);
      } else {
        sliceSizes = this.splits_.slice();
      }
      const outputsShape = [];
      for (const size of sliceSizes) {
        const outputShape = input.shape.slice();
        outputShape[axis] = size;
        outputsShape.push(outputShape);
      }
      for (let i = 0; i < outputs.length; ++i) {
        utils.checkShape(outputs[i].shape, outputsShape[i]);
      }
      this.needCheckOutputShape_ = false;
    }
    return outputs;
  }
}
