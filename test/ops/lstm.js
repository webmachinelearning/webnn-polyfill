'use strict';
import * as utils from '../utils.js';

describe('test lstm', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('lstmCell activations=[relu, relu, relu]', async () => {
    const builder = new MLGraphBuilder(context);
    const batchSize = 2;
    const inputSize = 2;
    const hiddenSize = 2;
    const input = builder.input(
        'input', {type: 'float32', dimensions: [batchSize, inputSize]});
    const weight = builder.constant(
        {
          type: 'float32',
          dimensions: [4 * hiddenSize, inputSize]},
        new Float32Array([
          1, -1, 2, -2, 1, -1, 2, -2,
          1, -1, 2, -2, 1, -1, 2, -2,
        ]));
    const recurrentWeight = builder.constant(
        {
          type: 'float32',
          dimensions: [4 * hiddenSize, hiddenSize]},
        new Float32Array(4 * hiddenSize * hiddenSize).fill(0.1));
    const hiddenState = builder.constant(
        {type: 'float32', dimensions: [batchSize, hiddenSize]},
        new Float32Array(batchSize * hiddenSize).fill(0));
    const cellState = builder.constant(
        {type: 'float32', dimensions: [batchSize, hiddenSize]},
        new Float32Array(batchSize * hiddenSize).fill(0));
    const bias = builder.constant(
        {type: 'float32', dimensions: [4 * hiddenSize]},
        new Float32Array([
          1, 2, 1, 2, 1, 2, 1, 2,
        ]));
    const recurrentBias = builder.constant(
        {type: 'float32', dimensions: [4 * hiddenSize]},
        new Float32Array([
          1, 2, 1, 2, 1, 2, 1, 2,
        ]));
    const peepholeWeight = builder.constant(
        {type: 'float32', dimensions: [3 * hiddenSize]},
        new Float32Array(3 * hiddenSize).fill(0));
    const activations = [
      builder.relu(),
      builder.relu(),
      builder.relu(),
    ];
    const operands = builder.lstmCell(
        input, weight, recurrentWeight, hiddenState, cellState, hiddenSize,
        {
          bias, recurrentBias, peepholeWeight, activations});
    const graph = await builder.build(
        {output0: operands[0], output1: operands[1]});
    const inputs = {'input': new Float32Array([1, 2, 2, 1])};
    const outputs = {
      'output0': new Float32Array(
          utils.sizeOfShape([batchSize, hiddenSize])),
      'output1': new Float32Array(
          utils.sizeOfShape([batchSize, hiddenSize])),
    };
    const result = await context.compute(graph, inputs, outputs);
    const expected = [
      [
        1, 8, 27, 216,
      ],
      [
        1, 4, 9, 36,
      ],
    ];
    for (let i = 0; i < expected.length; ++i) {
      utils.checkValue(result.outputs[`output${i}`], expected[i]);
    }
  });

  it('lstm returnSequence=true ' +
      'activations=[relu, relu, relu]', async () => {
    const builder = new MLGraphBuilder(context);
    const steps = 1;
    const batchSize = 2;
    const inputSize = 2;
    const hiddenSize = 2;
    const numDirections = 1;
    const input = builder.input(
        'input', {type: 'float32', dimensions: [steps, batchSize, inputSize]});
    const weight = builder.constant(
        {
          type: 'float32',
          dimensions: [numDirections, 4 * hiddenSize, inputSize]},
        new Float32Array([
          1, -1, 2, -2, 1, -1, 2, -2,
          1, -1, 2, -2, 1, -1, 2, -2,
        ]));
    const recurrentWeight = builder.constant(
        {
          type: 'float32',
          dimensions: [numDirections, 4 * hiddenSize, hiddenSize]},
        new Float32Array(4 * hiddenSize * hiddenSize).fill(0.1));
    const bias = builder.constant(
        {type: 'float32', dimensions: [numDirections, 4 * hiddenSize]},
        new Float32Array([
          1, 2, 1, 2, 1, 2, 1, 2,
        ]));
    const recurrentBias = builder.constant(
        {type: 'float32', dimensions: [numDirections, 4 * hiddenSize]},
        new Float32Array([
          1, 2, 1, 2, 1, 2, 1, 2,
        ]));
    const peepholeWeight = builder.constant(
        {type: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
        new Float32Array(3 * hiddenSize).fill(0));
    const initialHiddenState = builder.constant(
        {type: 'float32', dimensions: [numDirections, batchSize, hiddenSize]},
        new Float32Array(batchSize * hiddenSize).fill(0));
    const initialCellState = builder.constant(
        {type: 'float32', dimensions: [numDirections, batchSize, hiddenSize]},
        new Float32Array(batchSize * hiddenSize).fill(0));
    const returnSequence = true;
    const activations = [
      builder.relu(),
      builder.relu(),
      builder.relu(),
    ];
    const operands = builder.lstm(
        input, weight, recurrentWeight, steps, hiddenSize,
        {
          bias, recurrentBias, peepholeWeight, initialHiddenState,
          initialCellState, returnSequence, activations});
    const graph = await builder.build(
        {output0: operands[0], output1: operands[1], output2: operands[2]});
    const inputs = {'input': new Float32Array([1, 2, 2, 1])};
    const outputs = {
      'output0': new Float32Array(
          utils.sizeOfShape([numDirections, batchSize, hiddenSize])),
      'output1': new Float32Array(
          utils.sizeOfShape([numDirections, batchSize, hiddenSize])),
      'output2': new Float32Array(
          utils.sizeOfShape([steps, numDirections, batchSize, hiddenSize])),
    };
    const result = await context.compute(graph, inputs, outputs);
    const expected = [
      [
        1, 8, 27, 216,
      ],
      [
        1, 4, 9, 36,
      ],
      [
        1, 8, 27, 216,
      ],
    ];
    for (let i = 0; i < expected.length; ++i) {
      utils.checkValue(result.outputs[`output${i}`], expected[i]);
    }
  });

  it('lstm steps=2 direction="backward" returnSequence=true' +
      'activations=[relu, relu, relu]', async () => {
    const builder = new MLGraphBuilder(context);
    const steps = 2;
    const batchSize = 2;
    const inputSize = 2;
    const hiddenSize = 2;
    const numDirections = 1;
    const input = builder.input(
        'input', {type: 'float32', dimensions: [steps, batchSize, inputSize]});
    const weight = builder.constant(
        {
          type: 'float32',
          dimensions: [numDirections, 4 * hiddenSize, inputSize]},
        new Float32Array([
          1, -1, 2, -2, 1, -1, 2, -2,
          1, -1, 2, -2, 1, -1, 2, -2,
        ]));
    const recurrentWeight = builder.constant(
        {
          type: 'float32',
          dimensions: [numDirections, 4 * hiddenSize, hiddenSize]},
        new Float32Array(4 * hiddenSize * hiddenSize).fill(0.1));
    const bias = builder.constant(
        {type: 'float32', dimensions: [numDirections, 4 * hiddenSize]},
        new Float32Array([
          1, 2, 1, 2, 1, 2, 1, 2,
        ]));
    const recurrentBias = builder.constant(
        {type: 'float32', dimensions: [numDirections, 4 * hiddenSize]},
        new Float32Array([
          1, 2, 1, 2, 1, 2, 1, 2,
        ]));
    const peepholeWeight = builder.constant(
        {type: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
        new Float32Array(3 * hiddenSize).fill(0));
    const initialHiddenState = builder.constant(
        {type: 'float32', dimensions: [numDirections, batchSize, hiddenSize]},
        new Float32Array(batchSize * hiddenSize).fill(0));
    const initialCellState = builder.constant(
        {type: 'float32', dimensions: [numDirections, batchSize, hiddenSize]},
        new Float32Array(batchSize * hiddenSize).fill(0));
    const returnSequence = true;
    const direction = 'backward';
    const activations = [
      builder.relu(),
      builder.relu(),
      builder.relu(),
    ];
    const operands = builder.lstm(
        input, weight, recurrentWeight, steps, hiddenSize,
        {
          bias, recurrentBias, peepholeWeight, initialHiddenState,
          initialCellState, direction, returnSequence, activations});
    const graph = await builder.build(
        {output0: operands[0], output1: operands[1], output2: operands[2]});
    const inputs = {'input': new Float32Array([
      1, 2, 2, 1,
      3, 4, 1, 2,
    ])};
    const outputs = {
      'output0': new Float32Array(
          utils.sizeOfShape([numDirections, batchSize, hiddenSize])),
      'output1': new Float32Array(
          utils.sizeOfShape([numDirections, batchSize, hiddenSize])),
      'output2': new Float32Array(
          utils.sizeOfShape([steps, numDirections, batchSize, hiddenSize])),
    };
    const result = await context.compute(graph, inputs, outputs);
    const expected = [
      [
        10.46900082,  58.02900696,
        74.52900696, 518.94897461,
      ],
      [
        5.51000023, 20.01000214,
        19.11000061, 75.20999908,
      ],
      [
        1, 8,
        1, 8,
        10.46900082,  58.02900696,
        74.52900696, 518.94897461,
      ],
    ];
    for (let i = 0; i < expected.length; ++i) {
      utils.checkValue(result.outputs[`output${i}`], expected[i]);
    }
  });
});
