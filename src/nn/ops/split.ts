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
  private outputsShape_: number[][];

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
        options.axis === undefined || utils.isUnsignedInteger(options.axis),
        'The options.axis is invalid.');
    this.axis_ = options.axis ?? 0;

    // Prepare outputs.
    const numOutputs: number = utils.isInteger(splits) ? splits as number :
        (splits as number[]).length ;
    const axis = this.axis_;
    let sliceSizes = [];
    if (typeof splits === 'number') {
      sliceSizes =
          new Array(splits).fill(input.shape()[axis] / splits);
    } else {
      sliceSizes = splits.slice();
    }
    this.outputsShape_ = [];
    for (const size of sliceSizes) {
      const outputShape = input.shape().slice();
      outputShape[axis] = size;
      this.outputsShape_.push(outputShape);
    }

    for (let i = 0; i < numOutputs; ++i) {
      this.outputs.push(new OutputOperand(this,
          {dataType: this.input_.dataType(),
            dimensions: this.outputsShape_[i]}));
    }
  }

  inputs(): MLOperand[] {
    return [this.input_];
  }

  computeImpl(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor[] {
    const input: tf.Tensor = inputTensors.get(this.input_);
    const outputs = tf.split(input, this.splits_, this.axis_);
    if (this.needCheckOutputShape_) {
      for (let i = 0; i < outputs.length; ++i) {
        utils.checkShape(outputs[i].shape, this.outputsShape_[i]);
      }
      this.needCheckOutputShape_ = false;
    }
    return outputs;
  }
}
