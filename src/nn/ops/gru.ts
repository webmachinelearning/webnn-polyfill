import * as tf from '@tensorflow/tfjs-core';

import {MLGruCellOptions, MLGruOptions, MLRecurrentNetworkDirection, MLRecurrentNetworkWeightLayout} from '../graph_builder';
import {MLOperand, OutputOperand} from '../operand';
import {MLOperator, Operation, SingleOutputOperation} from '../operation';
import * as utils from '../utils';
import {UnaryMLOperator} from './unary';

export class Gru extends Operation {
  private input_: MLOperand;
  private weight_: MLOperand;
  private recurrentWeight_: MLOperand;
  private steps_: number;
  private hiddenSize_: number;
  private bias_?: MLOperand;
  private recurrentBias_?: MLOperand;
  private initialHiddenState_?: MLOperand;
  private resetAfter_: boolean;
  private returnSequence_: boolean;
  private direction_: MLRecurrentNetworkDirection;
  private layout_: MLRecurrentNetworkWeightLayout;
  private activations_: MLOperator[];
  private needCheckOutputShape_ = true;

  constructor(
      input: MLOperand, weight: MLOperand, recurrentWeight: MLOperand,
      steps: number, hiddenSize: number, options: MLGruOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    utils.validateOperand(weight);
    this.weight_ = weight;
    utils.validateOperand(recurrentWeight);
    this.recurrentWeight_ = recurrentWeight;
    utils.assert(
        utils.isInteger(steps) && steps > 0, 'The steps parameter is invalid.');
    this.steps_ = steps;
    utils.assert(
        utils.isInteger(hiddenSize) && hiddenSize > 0,
        'The hiddenSize parameter is invalid.');
    this.hiddenSize_ = hiddenSize;
    this.initOptions(
        options.bias, options.recurrentBias, options.initialHiddenState,
        options.resetAfter, options.returnSequence, options.direction,
        options.layout, options.activations);

    this.outputs.push(new OutputOperand(this));
    if (this.returnSequence_) {
      this.outputs_.push(new OutputOperand(this));
    }
  }

  initOptions(
      bias?: MLOperand, recurrentBias?: MLOperand,
      initialHiddenState?: MLOperand, resetAfter = true, returnSequence = false,
      direction:
          MLRecurrentNetworkDirection = MLRecurrentNetworkDirection.forward,
      layout:
          MLRecurrentNetworkWeightLayout = MLRecurrentNetworkWeightLayout.zrn,
      activations:
          MLOperator[] = [this.builder.sigmoid(), this.builder.tanh()]): void {
    utils.validateOptionalOperand(bias);
    this.bias_ = bias;
    utils.validateOptionalOperand(recurrentBias);
    this.recurrentBias_ = recurrentBias;
    utils.validateOptionalOperand(initialHiddenState);
    this.initialHiddenState_ = initialHiddenState;
    utils.assert(
        utils.isBoolean(resetAfter),
        'The resetAfter parameter is not a boolean.');
    this.resetAfter_ = resetAfter;
    utils.assert(
        utils.isBoolean(returnSequence),
        'The resetAfter parameter is not a boolean.');
    this.returnSequence_ = returnSequence;
    utils.assert(
        direction in MLRecurrentNetworkDirection,
        'The direction parameter is invalid.');
    this.direction_ = direction;
    utils.assert(
        layout in MLRecurrentNetworkWeightLayout,
        'The layout parameter is invalid.');
    this.layout_ = layout;
    utils.assert(
        activations instanceof Array && activations.length === 2 &&
            activations.every(a => a instanceof UnaryMLOperator),
        'The activations parameter is invalid.');
    this.activations_ = activations;
  }

  inputs(): MLOperand[] {
    const inputs: MLOperand[] =
        [this.input_, this.weight_, this.recurrentWeight_];
    if (this.bias_) {
      inputs.push(this.bias_);
    }
    if (this.recurrentBias_) {
      inputs.push(this.recurrentBias_);
    }
    if (this.initialHiddenState_) {
      inputs.push(this.initialHiddenState_);
    }
    return inputs;
  }

  computeImpl(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor[] {
    const input = inputTensors.get(this.input_);
    const weight = inputTensors.get(this.weight_);
    const recurrentWeight = inputTensors.get(this.recurrentWeight_);
    const bias = this.bias_ ? inputTensors.get(this.bias_) : undefined;
    const recurrentBias = this.recurrentWeight_ ?
        inputTensors.get(this.recurrentBias_) :
        undefined;
    const initialHiddenState = this.initialHiddenState_ ?
        inputTensors.get(this.initialHiddenState_) :
        undefined;
    const steps = this.steps_;
    const hiddenSize = this.hiddenSize_;
    const resetAfter = this.resetAfter_;
    const returnSequence = this.returnSequence_;
    const layout = this.layout_;
    const activations = this.activations_;
    const direction = this.direction_;

    const numDirections =
        (direction === MLRecurrentNetworkDirection.both ? 2 : 1);
    let hiddenState = initialHiddenState;

    if (hiddenState === undefined) {
      hiddenState = tf.zeros([numDirections, 1, hiddenSize]);
    }

    let sequence: tf.Tensor;
    const cellWeight: tf.Tensor[] = [];
    const cellRecurrentWeight: tf.Tensor[] = [];
    const cellBias: tf.Tensor[] = [];
    const cellRecurrentBias: tf.Tensor[] = [];

    for (let slot = 0; slot < numDirections; ++slot) {
      cellWeight.push(
          tf.squeeze(tf.slice(weight, [slot, 0, 0], [1, -1, -1]), [0]));
      cellRecurrentWeight.push(tf.squeeze(
          tf.slice(recurrentWeight, [slot, 0, 0], [1, -1, -1]), [0]));
      cellBias.push(
          bias ? (tf.squeeze(tf.slice(bias, [slot, 0], [1, -1]), [0])) :
                 undefined);
      cellRecurrentBias.push(
          recurrentBias ?
              (tf.squeeze(tf.slice(recurrentBias, [slot, 0], [1, -1]), [0])) :
              undefined);
    }

    for (let step = 0; step < steps; ++step) {
      const cellHidden: tf.Tensor[] = [];
      let cellOutput: tf.Tensor;

      for (let slot = 0; slot < numDirections; ++slot) {
        cellHidden.push(
            tf.squeeze(tf.slice(hiddenState, [slot, 0, 0], [1, -1, -1]), [0]));
      }

      for (let slot = 0; slot < numDirections; ++slot) {
        const slice =
            (slot === 1 || direction === MLRecurrentNetworkDirection.backward ?
                 steps - step - 1 :
                 step);
        const cellInput =
            tf.squeeze(tf.slice(input, [slice, 0, 0], [1, -1, -1]), [0]);

        const result = tf.reshape(
            GruCell.compute(
                cellInput, cellWeight[slot], cellRecurrentWeight[slot],
                cellHidden[slot], hiddenSize, activations, cellBias[slot],
                cellRecurrentBias[slot], resetAfter, layout),
            [1, -1, hiddenSize]);

        cellOutput = (cellOutput ? tf.concat([cellOutput, result], 0) : result);
      }

      hiddenState = cellOutput;

      if (returnSequence) {
        cellOutput = tf.reshape(cellOutput, [1, numDirections, -1, hiddenSize]);
        sequence =
            (sequence ? tf.concat([sequence, cellOutput], 0) : cellOutput);
      }
    }
    const outputs = [hiddenState];
    if (returnSequence) {
      outputs.push(sequence);
    }
    if (this.needCheckOutputShape_) {
      // the first output is 3-D tensor of shape
      //   [num_directions, batch_size, hidden_size]
      const outputsShape = [[numDirections, input.shape[1], hiddenSize]];
      if (returnSequence) {
        // returnSequence true, the second output tensor of shape
        //   [steps, num_directions, batch_size, hidden_size]
        outputsShape.push([steps, numDirections, input.shape[1], hiddenSize]);
      }
      for (let i = 0; i < outputs.length; ++i) {
        utils.checkShape(outputs[i].shape, outputsShape[i]);
      }
      this.needCheckOutputShape_ = false;
    }
    return outputs;
  }
}

export class GruCell extends SingleOutputOperation {
  private input_: MLOperand;
  private weight_: MLOperand;
  private recurrentWeight_: MLOperand;
  private hiddenState_: MLOperand;
  private hiddenSize_: number;
  private bias_?: MLOperand;
  private recurrentBias_?: MLOperand;
  private resetAfter_: boolean;
  private layout_: MLRecurrentNetworkWeightLayout;
  private activations_: MLOperator[];
  private needCheckOutputShape_ = true;

  constructor(
      input: MLOperand, weight: MLOperand, recurrentWeight: MLOperand,
      hiddenState: MLOperand, hiddenSize: number,
      options: MLGruCellOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    utils.validateOperand(weight);
    this.weight_ = weight;
    utils.validateOperand(recurrentWeight);
    this.recurrentWeight_ = recurrentWeight;
    utils.validateOperand(hiddenState);
    this.hiddenState_ = hiddenState;
    utils.assert(
        utils.isInteger(hiddenSize) && hiddenSize > 0,
        'The hiddenSize parameter is invalid.');
    this.hiddenSize_ = hiddenSize;
    this.initOptions(
        options.bias, options.recurrentBias, options.resetAfter, options.layout,
        options.activations);
  }

  private initOptions(
      bias?: MLOperand, recurrentBias?: MLOperand, resetAfter = true,
      layout:
          MLRecurrentNetworkWeightLayout = MLRecurrentNetworkWeightLayout.zrn,
      activations:
          MLOperator[] = [this.builder.sigmoid(), this.builder.tanh()]) {
    utils.validateOptionalOperand(bias);
    this.bias_ = bias;
    utils.validateOptionalOperand(recurrentBias);
    this.recurrentBias_ = recurrentBias;
    utils.assert(
        utils.isBoolean(resetAfter),
        'The resetAfter parameter is not a boolean.');
    this.resetAfter_ = resetAfter;
    utils.assert(
        layout in MLRecurrentNetworkWeightLayout,
        'The layout parameter is invalid.');
    this.layout_ = layout;
    utils.assert(
        activations instanceof Array && activations.length === 2 &&
            activations.every(a => a instanceof UnaryMLOperator),
        'The activations parameter is invalid.');
    this.activations_ = activations;
  }

  inputs(): MLOperand[] {
    const inputs: MLOperand[] =
        [this.input_, this.weight_, this.recurrentWeight_, this.hiddenState_];
    if (this.bias_) {
      inputs.push(this.bias_);
    }
    if (this.recurrentBias_) {
      inputs.push(this.recurrentBias_);
    }
    return inputs;
  }

  static compute(
      input: tf.Tensor, weight: tf.Tensor, recurrentWeight: tf.Tensor,
      hiddenState: tf.Tensor, hiddenSize: number, activations: MLOperator[],
      bias?: tf.Tensor, recurrentBias?: tf.Tensor, resetAfter = true,
      layout:
          MLRecurrentNetworkWeightLayout = MLRecurrentNetworkWeightLayout.zrn):
      tf.Tensor {
    const one = tf.scalar(1);
    const zero = tf.scalar(0);
    const starts = layout === MLRecurrentNetworkWeightLayout.zrn ?
        {z: 0, r: hiddenSize, n: 2 * hiddenSize} :
        /*rzn*/ {r: 0, z: hiddenSize, n: 2 * hiddenSize};
    const activation0: UnaryMLOperator = activations[0] as UnaryMLOperator;
    const activation1: UnaryMLOperator = activations[1] as UnaryMLOperator;
    // update gate
    const z = activation0.runOp(tf.add(
        tf.add(
            (bias ? tf.slice(bias, [starts.z], [hiddenSize]) : zero),
            (recurrentBias ? tf.slice(recurrentBias, [starts.z], [hiddenSize]) :
                             zero)),
        tf.add(
            tf.matMul(
                input,
                tf.transpose(
                    tf.slice(weight, [starts.z, 0], [hiddenSize, -1]))),
            tf.matMul(
                hiddenState,
                tf.transpose(tf.slice(
                    recurrentWeight, [starts.z, 0], [hiddenSize, -1]))))));
    // reset gate
    const r = activation0.runOp(tf.add(
        tf.add(
            (bias ? tf.slice(bias, [starts.r], [hiddenSize]) : zero),
            (recurrentBias ? tf.slice(recurrentBias, [starts.r], [hiddenSize]) :
                             zero)),
        tf.add(
            tf.matMul(
                input,
                tf.transpose(
                    tf.slice(weight, [starts.r, 0], [hiddenSize, -1]))),
            tf.matMul(
                hiddenState,
                tf.transpose(tf.slice(
                    recurrentWeight, [starts.r, 0], [hiddenSize, -1]))))));
    // new gate
    let n;
    if (resetAfter) {
      n = activation1.runOp(tf.add(
          (bias ? tf.slice(bias, [starts.n], [hiddenSize]) : zero),
          tf.add(
              tf.matMul(
                  input,
                  tf.transpose(
                      tf.slice(weight, [starts.n, 0], [hiddenSize, -1]))),
              tf.mul(
                  r,
                  tf.add(
                      (recurrentBias ?
                           tf.slice(recurrentBias, [starts.n], [hiddenSize]) :
                           zero),
                      tf.matMul(
                          hiddenState,
                          tf.transpose(tf.slice(
                              recurrentWeight, [starts.n, 0],
                              [hiddenSize, -1]))))))));
    } else {
      n = activation1.runOp(tf.add(
          tf.add(
              (bias ? tf.slice(bias, [starts.n], [hiddenSize]) : zero),
              (recurrentBias ?
                   tf.slice(recurrentBias, [starts.n], [hiddenSize]) :
                   zero)),
          tf.add(
              tf.matMul(
                  input,
                  tf.transpose(
                      tf.slice(weight, [starts.n, 0], [hiddenSize, -1]))),
              tf.matMul(
                  tf.mul(r, hiddenState),
                  tf.transpose(tf.slice(
                      recurrentWeight, [starts.n, 0], [hiddenSize, -1]))))));
    }
    // compute the new hidden state
    return tf.add(tf.mul(z, hiddenState), tf.mul(n, tf.sub(one, z)));
  }

  run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor {
    const input: tf.Tensor = inputTensors.get(this.input_);
    const output = GruCell.compute(
        input, inputTensors.get(this.weight_),
        inputTensors.get(this.recurrentWeight_),
        inputTensors.get(this.hiddenState_), this.hiddenSize_,
        this.activations_,
        this.bias_ ? inputTensors.get(this.bias_) : undefined,
        this.recurrentBias_ ? inputTensors.get(this.recurrentBias_) : undefined,
        this.resetAfter_, this.layout_);
    if (this.needCheckOutputShape_) {
      // output shape [batch_size, hidden_size]
      const outputShape = [input.shape[0], this.hiddenSize_];
      utils.checkShape(output.shape, outputShape);
      this.needCheckOutputShape_ = false;
    }
    return output;
  }
}
