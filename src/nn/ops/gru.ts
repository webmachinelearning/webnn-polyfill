import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from '../compilation';
import {GruCellOptions, GruOptions, RecurrentNetworkActivation, RecurrentNetworkDirection, RecurrentNetworkWeightLayout} from '../model_builder';
import {Operand, OutputOperand} from '../operand';
import {Operation, SingleOutputOperation} from '../operation';
import * as utils from '../utils';

export class Gru extends Operation {
  private input_: Operand;
  private weight_: Operand;
  private recurrentWeight_: Operand;
  private steps_: number;
  private hiddenSize_: number;
  private bias_?: Operand;
  private recurrentBias_?: Operand;
  private initialHiddenState_?: Operand;
  private resetAfter_: boolean;
  private returnSequence_: boolean;
  private direction_: RecurrentNetworkDirection;
  private layout_: RecurrentNetworkWeightLayout;
  private activations_: RecurrentNetworkActivation[];

  constructor(
      input: Operand, weight: Operand, recurrentWeight: Operand, steps: number,
      hiddenSize: number, options: GruOptions = {}) {
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
      bias?: Operand, recurrentBias?: Operand, initialHiddenState?: Operand,
      resetAfter = true, returnSequence = false,
      direction: RecurrentNetworkDirection = RecurrentNetworkDirection.forward,
      layout: RecurrentNetworkWeightLayout = RecurrentNetworkWeightLayout.zrn,
      activations: RecurrentNetworkActivation[] = [
        RecurrentNetworkActivation.sigmoid, RecurrentNetworkActivation.tanh
      ]): void {
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
        direction in RecurrentNetworkDirection,
        'The direction parameter is invalid.');
    this.direction_ = direction;
    utils.assert(
        layout in RecurrentNetworkWeightLayout,
        'The layout parameter is invalid.');
    this.layout_ = layout;
    utils.assert(
        activations instanceof Array && activations.length === 2 &&
            activations.every(a => a in RecurrentNetworkActivation),
        'The activations parameter is invalid.');
    this.activations_ = activations;
  }

  inputs(): Operand[] {
    const inputs: Operand[] =
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

  computeImpl(context: ExecutionContext): tf.Tensor[] {
    const input = context.getTensor(this.input_);
    const weight = context.getTensor(this.weight_);
    const recurrentWeight = context.getTensor(this.recurrentWeight_);
    const bias = this.bias_ ? context.getTensor(this.bias_) : undefined;
    const recurrentBias = this.recurrentWeight_ ?
        context.getTensor(this.recurrentBias_) :
        undefined;
    const initialHiddenState = this.initialHiddenState_ ?
        context.getTensor(this.initialHiddenState_) :
        undefined;
    const steps = this.steps_;
    const hiddenSize = this.hiddenSize_;
    const resetAfter = this.resetAfter_;
    const returnSequence = this.returnSequence_;
    const layout = this.layout_;
    const activations = this.activations_;
    const direction = this.direction_;

    const numDirections =
        (direction === RecurrentNetworkDirection.both ? 2 : 1);
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
            (slot === 1 || direction === RecurrentNetworkDirection.backward ?
                 steps - step - 1 :
                 step);
        const cellInput =
            tf.squeeze(tf.slice(input, [slice, 0, 0], [1, -1, -1]), [0]);

        const result = tf.reshape(
            GruCell.compute(
                cellInput, cellWeight[slot], cellRecurrentWeight[slot],
                cellHidden[slot], hiddenSize, cellBias[slot],
                cellRecurrentBias[slot], resetAfter, layout, activations),
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

    return [hiddenState, sequence];
  }

  compute(context: ExecutionContext): void {
    const outputTensors = this.computeImpl(context);
    context.setOutputTensor(this.outputs[0], outputTensors[0]);
    if (this.returnSequence_) {
      context.setOutputTensor(this.outputs[1], outputTensors[1]);
    }
  }
}

export class GruCell extends SingleOutputOperation {
  private input_: Operand;
  private weight_: Operand;
  private recurrentWeight_: Operand;
  private hiddenState_: Operand;
  private hiddenSize_: number;
  private bias_?: Operand;
  private recurrentBias_?: Operand;
  private resetAfter_: boolean;
  private layout_: RecurrentNetworkWeightLayout;
  private activations_: RecurrentNetworkActivation[];

  constructor(
      input: Operand, weight: Operand, recurrentWeight: Operand,
      hiddenState: Operand, hiddenSize: number, options: GruCellOptions = {}) {
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
      bias?: Operand, recurrentBias?: Operand, resetAfter = true,
      layout: RecurrentNetworkWeightLayout = RecurrentNetworkWeightLayout.zrn,
      activations: RecurrentNetworkActivation[] = [
        RecurrentNetworkActivation.sigmoid, RecurrentNetworkActivation.tanh
      ]) {
    utils.validateOptionalOperand(bias);
    this.bias_ = bias;
    utils.validateOptionalOperand(recurrentBias);
    this.recurrentBias_ = recurrentBias;
    utils.assert(
        utils.isBoolean(resetAfter),
        'The resetAfter parameter is not a boolean.');
    this.resetAfter_ = resetAfter;
    utils.assert(
        layout in RecurrentNetworkWeightLayout,
        'The layout parameter is invalid.');
    this.layout_ = layout;
    utils.assert(
        activations instanceof Array && activations.length === 2 &&
            activations.every(a => a in RecurrentNetworkActivation),
        'The activations parameter is invalid.');
    this.activations_ = activations;
  }

  inputs(): Operand[] {
    const inputs: Operand[] =
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
      hiddenState: tf.Tensor, hiddenSize: number, bias?: tf.Tensor,
      recurrentBias?: tf.Tensor, resetAfter = true,
      layout: RecurrentNetworkWeightLayout = RecurrentNetworkWeightLayout.zrn,
      activations: RecurrentNetworkActivation[] = [
        RecurrentNetworkActivation.sigmoid, RecurrentNetworkActivation.tanh
      ]): tf.Tensor {
    const one = tf.scalar(1);
    const zero = tf.scalar(0);
    const starts = layout === RecurrentNetworkWeightLayout.zrn ?
        {z: 0, r: hiddenSize, n: 2 * hiddenSize} :
        /*rzn*/ {r: 0, z: hiddenSize, n: 2 * hiddenSize};
    // update gate
    const z = tf[activations[0]](tf.add(
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
    const r = tf[activations[0]](tf.add(
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
      n = tf[activations[1]](tf.add(
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
      n = tf[activations[1]](tf.add(
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

  run(context: ExecutionContext): tf.Tensor {
    return GruCell.compute(
        context.getTensor(this.input_), context.getTensor(this.weight_),
        context.getTensor(this.recurrentWeight_),
        context.getTensor(this.hiddenState_), this.hiddenSize_,
        this.bias_ ? context.getTensor(this.bias_) : undefined,
        this.recurrentBias_ ? context.getTensor(this.recurrentBias_) :
                              undefined,
        this.resetAfter_, this.layout_, this.activations_);
  }
}
