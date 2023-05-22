import * as tf from '@tensorflow/tfjs-core';

import {MLLstmCellOptions, MLLstmOptions, MLLstmWeightLayout, MLRecurrentNetworkDirection} from '../graph_builder';
import {MLOperand, OutputOperand} from '../operand';
import {MLActivation, Operation} from '../operation';
import * as utils from '../utils';
import {UnaryMLActivation} from './unary';

export class Lstm extends Operation {
  private input_: MLOperand;
  private weight_: MLOperand;
  private recurrentWeight_: MLOperand;
  private steps_: number;
  private hiddenSize_: number;
  private bias_?: MLOperand;
  private recurrentBias_?: MLOperand;
  private peepholeWeight_?: MLOperand;
  private initialHiddenState_?: MLOperand;
  private initialCellState_?: MLOperand;
  private returnSequence_: boolean;
  private direction_: MLRecurrentNetworkDirection;
  private layout_: MLLstmWeightLayout;
  private activations_: MLActivation[];
  private needCheckOutputShape_ = true;

  constructor(
      input: MLOperand, weight: MLOperand, recurrentWeight: MLOperand,
      steps: number, hiddenSize: number, options: MLLstmOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    utils.validateOperand(weight);
    this.weight_ = weight;
    utils.validateOperand(recurrentWeight);
    this.recurrentWeight_ = recurrentWeight;
    utils.assert(
        utils.isUnsignedInteger(steps) && steps > 0,
        'The steps parameter is invalid.');
    this.steps_ = steps;
    utils.assert(
        utils.isUnsignedInteger(hiddenSize) && hiddenSize > 0,
        'The hiddenSize parameter is invalid.');
    this.hiddenSize_ = hiddenSize;

    this.initOptions(
        options.bias, options.recurrentBias, options.peepholeWeight,
        options.initialHiddenState, options.initialCellState,
        options.returnSequence, options.direction,
        options.layout, options.activations);

    this.outputs.push(new OutputOperand(this));
    this.outputs.push(new OutputOperand(this));
    if (this.returnSequence_) {
      this.outputs.push(new OutputOperand(this));
    }
  }

  initOptions(
      bias?: MLOperand, recurrentBias?: MLOperand,
      peepholeWeight?: MLOperand, initialHiddenState?: MLOperand,
      initialCellState?: MLOperand, returnSequence = false,
      direction:
          MLRecurrentNetworkDirection = MLRecurrentNetworkDirection.forward,
      layout: MLLstmWeightLayout = MLLstmWeightLayout.iofg,
      activations:
          MLActivation[] = [
              this.builder.sigmoid(), this.builder.tanh(), this.builder.tanh(),
          ]):
  void {
    utils.validateOptionalOperand(bias);
    this.bias_ = bias;
    utils.validateOptionalOperand(recurrentBias);
    this.recurrentBias_ = recurrentBias;
    utils.validateOptionalOperand(peepholeWeight);
    this.peepholeWeight_ = peepholeWeight;
    utils.validateOptionalOperand(initialHiddenState);
    this.initialHiddenState_ = initialHiddenState;
    utils.validateOptionalOperand(initialCellState);
    this.initialCellState_ = initialCellState;
    utils.assert(
        utils.isBoolean(returnSequence),
        'The returnSequence parameter is not a boolean.');
    this.returnSequence_ = returnSequence;
    utils.assert(
        direction in MLRecurrentNetworkDirection,
        'The direction parameter is invalid.');
    this.direction_ = direction;
    utils.assert(
        layout in MLLstmWeightLayout,
        'The layout parameter is invalid.');
    this.layout_ = layout;
    utils.assert(
        activations instanceof Array && activations.length === 3 &&
            activations.every(a => a instanceof UnaryMLActivation),
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
    if (this.peepholeWeight_) {
      inputs.push(this.peepholeWeight_);
    }
    if (this.initialHiddenState_) {
      inputs.push(this.initialHiddenState_);
    }
    if (this.initialCellState_) {
      inputs.push(this.initialCellState_);
    }
    return inputs;
  }

  computeImpl(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor[] {
    const input = inputTensors.get(this.input_);
    // The input 3-D tensor of shape [steps, batch_size, inputSize]
    const [, batchSize, inputSize] = input.shape;
    const weight = inputTensors.get(this.weight_);
    const recurrentWeight = inputTensors.get(this.recurrentWeight_);
    const bias = this.bias_ ? inputTensors.get(this.bias_) : undefined;
    const recurrentBias = this.recurrentWeight_ ?
        inputTensors.get(this.recurrentBias_) :
        undefined;
    const peepholeWeight = this.peepholeWeight_ ?
        inputTensors.get(this.peepholeWeight_) :
        undefined; 
    const initialHiddenState = this.initialHiddenState_ ?
        inputTensors.get(this.initialHiddenState_) :
        undefined;
    const initialCellState = this.initialCellState_ ?
        inputTensors.get(this.initialCellState_) :
        undefined;
    const steps = this.steps_;
    const hiddenSize = this.hiddenSize_;
    const returnSequence = this.returnSequence_;
    const layout = this.layout_;
    const activations = this.activations_;
    const direction = this.direction_;

    const numDirections =
        (direction === MLRecurrentNetworkDirection.both ? 2 : 1);
    let hiddenState = initialHiddenState;
    let cellState = initialCellState;

    if (hiddenState === undefined) {
      hiddenState = tf.zeros([numDirections, 1, hiddenSize]);
    }

    if (cellState === undefined) {
      cellState = tf.zeros([numDirections, 1, hiddenSize]);
    }

    let sequence: tf.Tensor;
    const currentWeight: tf.Tensor[] = [];
    const currentRecurrentWeight: tf.Tensor[] = [];
    const currentBias: tf.Tensor[] = [];
    const currentRecurrentBias: tf.Tensor[] = [];
    const currentPeepholeWeight: tf.Tensor[] = [];

    for (let dir = 0; dir < numDirections; ++dir) {
      currentWeight.push(
          tf.squeeze(tf.slice(
            weight, [dir, 0, 0], [1, 4 * hiddenSize, inputSize]), [0]));
      currentRecurrentWeight.push(tf.squeeze(
          tf.slice(
              recurrentWeight, [dir, 0, 0], [1, 4 * hiddenSize, hiddenSize]),
              [0]));
      currentBias.push(bias ?
          (tf.squeeze(tf.slice(bias, [dir, 0], [1, 4 * hiddenSize]), [0])) :
          undefined);
      currentRecurrentBias.push(recurrentBias ?
          (tf.squeeze(
              tf.slice(recurrentBias, [dir, 0], [1, 4 * hiddenSize]), [0])) :
          undefined);
      currentPeepholeWeight.push(peepholeWeight ?
          (tf.squeeze(
              tf.slice(peepholeWeight, [dir, 0], [1, 3 * hiddenSize]), [0])) :
          undefined);
    }

    for (let step = 0; step < steps; ++step) {
      const currentHidden: tf.Tensor[] = [];
      const currentCell: tf.Tensor[] = [];
      let nextHidden: tf.Tensor;
      let nextCell: tf.Tensor;

      for (let dir = 0; dir < numDirections; ++dir) {
        currentHidden.push(tf.squeeze(
            tf.slice(
                hiddenState, [dir, 0, 0], [1, batchSize, hiddenSize]), [0]));
        currentCell.push(tf.squeeze(
            tf.slice(
                cellState, [dir, 0, 0], [1, batchSize, hiddenSize]), [0]));
      }

      for (let dir = 0; dir < numDirections; ++dir) {
        const slice =
            (dir === 1 || direction === MLRecurrentNetworkDirection.backward ?
                 steps - step - 1 :
                 step);
        const currentInput = tf.squeeze(
            tf.slice(input, [slice, 0, 0], [1, batchSize, inputSize]), [0]);

        const results = LstmCell.compute(
                currentInput, currentWeight[dir], currentRecurrentWeight[dir],
                currentHidden[dir], currentCell[dir], hiddenSize, activations,
                currentBias[dir], currentRecurrentBias[dir],
                currentPeepholeWeight[dir], layout);
        const output = tf.reshape(results[0], [1, -1, hiddenSize]);
        const cell = tf.reshape(results[1], [1, -1, hiddenSize]);
        nextHidden = (nextHidden ? tf.concat([nextHidden, output], 0) : output);
        nextCell = (nextCell ? tf.concat([nextCell, cell], 0) : cell);
      }

      hiddenState = nextHidden;
      cellState = nextCell;

      if (returnSequence) {
        nextHidden = tf.reshape(nextHidden, [1, numDirections, -1, hiddenSize]);
        sequence =
            (sequence ? tf.concat([sequence, nextHidden], 0) : nextHidden);
      }
    }
    const outputs = [hiddenState, cellState];
    if (returnSequence) {
      outputs.push(sequence);
    }
    if (this.needCheckOutputShape_) {
      // the first output is 3D tensor of shape
      //   [num_directions, batch_size, hiddenSize]
      // the second output is 3D tensor of shape
      //   [num_directions, batch_size, hiddenSize]
      const outputsShape = [
        [numDirections, batchSize, hiddenSize],
        [numDirections, batchSize, hiddenSize],
      ];
      if (returnSequence) {
        // returnSequence true, the third 4D output tensor of shape
        //   [steps, num_directions, batch_size, hiddenSize]
        outputsShape.push([steps, numDirections, batchSize, hiddenSize]);
      }
      for (let i = 0; i < outputs.length; ++i) {
        utils.checkShape(outputs[i].shape, outputsShape[i]);
      }
      this.needCheckOutputShape_ = false;
    }
    return outputs;
  }
}

export class LstmCell extends Operation {
  private input_: MLOperand;
  private weight_: MLOperand;
  private recurrentWeight_: MLOperand;
  private hiddenState_: MLOperand;
  private cellState_: MLOperand;
  private hiddenSize_: number;
  private bias_?: MLOperand;
  private recurrentBias_?: MLOperand;
  private peepholeWeight_?: MLOperand;
  private layout_: MLLstmWeightLayout;
  private activations_: MLActivation[];
  private needCheckOutputShape_ = true;

  constructor(
      input: MLOperand, weight: MLOperand, recurrentWeight: MLOperand,
      hiddenState: MLOperand, cellState: MLOperand, hiddenSize: number,
      options: MLLstmCellOptions = {}) {
    super(input.builder);
    utils.validateOperand(input);
    this.input_ = input;
    utils.validateOperand(weight);
    this.weight_ = weight;
    utils.validateOperand(recurrentWeight);
    this.recurrentWeight_ = recurrentWeight;
    utils.validateOperand(hiddenState);
    this.hiddenState_ = hiddenState;
    utils.validateOperand(cellState);
    this.cellState_ = cellState;
    utils.assert(
        utils.isUnsignedInteger(hiddenSize) && hiddenSize > 0,
        'The hiddenSize parameter is invalid.');
    this.hiddenSize_ = hiddenSize;

    this.initOptions(
        options.bias, options.recurrentBias, options.peepholeWeight,
        options.layout, options.activations);
    this.outputs.push(new OutputOperand(this));
    this.outputs.push(new OutputOperand(this));
  }

  private initOptions(
      bias?: MLOperand, recurrentBias?: MLOperand, peepholeWeight?: MLOperand,
      layout: MLLstmWeightLayout = MLLstmWeightLayout.iofg,
      activations:
          MLActivation[] = [
              this.builder.sigmoid(), this.builder.tanh(), this.builder.tanh(),
          ]) {
    utils.validateOptionalOperand(bias);
    this.bias_ = bias;
    utils.validateOptionalOperand(recurrentBias);
    this.recurrentBias_ = recurrentBias;
    utils.validateOptionalOperand(peepholeWeight);
    this.peepholeWeight_ = peepholeWeight;
    utils.assert(
        layout in MLLstmWeightLayout,
        'The layout parameter is invalid.');
    this.layout_ = layout;
    utils.assert(
        activations instanceof Array && activations.length === 3 &&
            activations.every(a => a instanceof UnaryMLActivation),
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
    if (this.peepholeWeight_) {
      inputs.push(this.peepholeWeight_);
    }
    if (this.hiddenState_) {
      inputs.push(this.hiddenState_);
    }
    if (this.cellState_) {
      inputs.push(this.cellState_);
    }
    return inputs;
  }

  static compute(
      input: tf.Tensor, weight: tf.Tensor, recurrentWeight: tf.Tensor,
      hiddenState: tf.Tensor, cellState: tf.Tensor, hiddenSize: number,
      activations: MLActivation[], bias?: tf.Tensor, recurrentBias?: tf.Tensor,
      peepholeWeight?: tf.Tensor, 
      layout:
          MLLstmWeightLayout = MLLstmWeightLayout.iofg):
      tf.Tensor[] {
        // The input 2-D tensor of shape [batch_size, inputSize]
        const inputSize = input.shape[1];
        const activation0: UnaryMLActivation =
            activations[0] as UnaryMLActivation;
        const activation1: UnaryMLActivation =
            activations[1] as UnaryMLActivation;
        const activation2: UnaryMLActivation =
            activations[2] as UnaryMLActivation;
        const zero = tf.scalar(0);
        const starts = layout === MLLstmWeightLayout.iofg ?
        {i: 0, o: hiddenSize, f: 2 * hiddenSize, g: 3 * hiddenSize} :
        /*ifgo*/ {i: 0, f: hiddenSize, g: 2 * hiddenSize, 0: 3 * hiddenSize};

        // input gate (i)
        const i = activation0.runOp(tf.add(
            tf.mul(
              cellState,
              // The pack ordering of the weight vectors is for the 
              // input (i), output (o), and forget (f) gate respectively.
              (peepholeWeight ?
                  tf.slice(peepholeWeight, [0], [hiddenSize]) : zero)),
            tf.add(
              tf.add(
                (bias ? tf.slice(bias, [starts.i], [hiddenSize]) : zero),
                (recurrentBias ?
                    tf.slice(recurrentBias, [starts.i], [hiddenSize]) : zero)),
              tf.add(
                tf.matMul(
                  input,
                  tf.transpose(tf.slice(
                      weight, [starts.i, 0], [hiddenSize, inputSize]))),
                tf.matMul(
                  hiddenState,
                  tf.transpose(tf.slice(
                      recurrentWeight, [starts.i, 0],
                      [hiddenSize, hiddenSize])))))));
        
        // forget gate (f)
        const f = activation0.runOp(tf.add(
            tf.mul(
              cellState,
              (peepholeWeight ?
                  tf.slice(peepholeWeight, [2 * hiddenSize], [hiddenSize]) :
                  zero)),
            tf.add(
              tf.add(
                (bias ? tf.slice(bias, [starts.f], [hiddenSize]) : zero),
                (recurrentBias ?
                    tf.slice(recurrentBias, [starts.f], [hiddenSize]) :
                    zero)),
              tf.add(
                tf.matMul(
                  input,
                  tf.transpose(tf.slice(
                    weight, [starts.f, 0], [hiddenSize, inputSize]))),
                tf.matMul(
                  hiddenState,
                  tf.transpose(tf.slice(
                      recurrentWeight, [starts.f, 0],
                      [hiddenSize, hiddenSize])))))));
        
        // cell gate (g)
        const g = activation1.runOp(tf.add(
            tf.add(
              (bias ? tf.slice(bias, [starts.g], [hiddenSize]) : zero),
              (recurrentBias ?
                  tf.slice(recurrentBias, [starts.g], [hiddenSize]) :
                  zero)),
            tf.add(
              tf.matMul(
                input,
                tf.transpose(tf.slice(
                    weight, [starts.g, 0], [hiddenSize, inputSize]))),
              tf.matMul(
                hiddenState,
                tf.transpose(tf.slice(
                    recurrentWeight, [starts.g, 0],
                    [hiddenSize, hiddenSize]))))));
        
        // output gate (o)
        const o = activation0.runOp(tf.add(
            tf.mul(
              cellState,
              (peepholeWeight ?
                  tf.slice(peepholeWeight, [hiddenSize], [hiddenSize]) :
                  zero)),
            tf.add(
              tf.add(
                (bias ? tf.slice(bias, [starts.o], [hiddenSize]) : zero),
                (recurrentBias ?
                    tf.slice(recurrentBias, [starts.o], [hiddenSize]) :
                    zero)),
              tf.add(
                tf.matMul(
                  input,
                  tf.transpose(tf.slice(
                      weight, [starts.o, 0], [hiddenSize, inputSize]))),
                tf.matMul(
                  hiddenState,
                  tf.transpose(tf.slice(
                      recurrentWeight, [starts.o, 0],
                      [hiddenSize, hiddenSize])))))));
        
        // output cell state (ct)
        const ct = tf.add(tf.mul(f, cellState), tf.mul(i, g));
        
        // output hidden state (ht)
        const ht = tf.mul(o, activation2.runOp(ct));
        
        return [ht, ct];
  }

  computeImpl(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor[] {
    const input: tf.Tensor = inputTensors.get(this.input_);
    const batchSize = input.shape[0];
    const outputs = LstmCell.compute(
        input, inputTensors.get(this.weight_),
        inputTensors.get(this.recurrentWeight_),
        inputTensors.get(this.hiddenState_),
        inputTensors.get(this.cellState_),
        this.hiddenSize_, this.activations_,
        this.bias_ ? inputTensors.get(this.bias_) : undefined,
        this.recurrentBias_ ? inputTensors.get(this.recurrentBias_) :
          undefined,
        this.peepholeWeight_ ? inputTensors.get(this.peepholeWeight_) :
          undefined,
        this.layout_);
    if (this.needCheckOutputShape_) {
      // Both are 2-D tensors of shape [batch_size, hidden_size].
      const outputsShape = [
          [batchSize, this.hiddenSize_], [batchSize, this.hiddenSize_],
      ];
      for (let i = 0; i < outputs.length; ++i) {
        utils.checkShape(outputs[i].shape, outputsShape[i]);
      }
      this.needCheckOutputShape_ = false;
    }
    return outputs;
  }
}
