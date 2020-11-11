'use strict';
import * as utils from '../utils.js';

describe('test gru', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('gruCell defaults', async function() {
    const builder = nn.createModelBuilder();
    const batchSize = 3;
    const inputSize = 2;
    const hiddenSize = 5;
    const input = builder.input(
        'input', {type: 'float32', dimensions: [batchSize, inputSize]});
    const weight = builder.constant(
        {type: 'float32', dimensions: [3 * hiddenSize, inputSize]},
        new Float32Array(3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = builder.constant(
        {type: 'float32', dimensions: [3 * hiddenSize, hiddenSize]},
        new Float32Array(3 * hiddenSize * hiddenSize).fill(0.1));
    const hiddenState = builder.constant(
        {type: 'float32', dimensions: [batchSize, hiddenSize]},
        new Float32Array(batchSize * hiddenSize).fill(0));
    const output = builder.gruCell(
        input, weight, recurrentWeight, hiddenState, hiddenSize);
    const model = builder.createModel({output});
    const compiledModel = await model.compile();
    const inputs = {'input': {buffer: new Float32Array([1, 2, 3, 4, 5, 6])}};
    const outputs = await compiledModel.compute(inputs);
    utils.checkShape(outputs.output.dimensions, [batchSize, hiddenSize]);
    const expected = [
      0.12397027,
      0.12397027,
      0.12397027,
      0.12397027,
      0.12397027,
      0.20053662,
      0.20053662,
      0.20053662,
      0.20053662,
      0.20053662,
      0.19991654,
      0.19991654,
      0.19991654,
      0.19991654,
      0.19991654,
    ];
    utils.checkValue(outputs.output.buffer, expected);
  });

  it('gruCell with bias', async function() {
    const builder = nn.createModelBuilder();
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 3;
    const input = builder.input(
        'input', {type: 'float32', dimensions: [batchSize, inputSize]});
    const weight = builder.constant(
        {type: 'float32', dimensions: [3 * hiddenSize, inputSize]},
        new Float32Array(3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = builder.constant(
        {type: 'float32', dimensions: [3 * hiddenSize, hiddenSize]},
        new Float32Array(3 * hiddenSize * hiddenSize).fill(0.1));
    const hiddenState = builder.constant(
        {type: 'float32', dimensions: [batchSize, hiddenSize]},
        new Float32Array(batchSize * hiddenSize).fill(0));
    const bias = builder.constant(
        {type: 'float32', dimensions: [3 * hiddenSize]},
        new Float32Array(3 * hiddenSize).fill(0.1));
    const recurrentBias = builder.constant(
        {type: 'float32', dimensions: [3 * hiddenSize]},
        new Float32Array(3 * hiddenSize).fill(0));
    const output = builder.gruCell(
        input, weight, recurrentWeight, hiddenState, hiddenSize,
        {bias, recurrentBias});
    const model = builder.createModel({output});
    const compiledModel = await model.compile();
    const inputs = {
      'input': {buffer: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9])},
    };
    const outputs = await compiledModel.compute(inputs);
    utils.checkShape(outputs.output.dimensions, [batchSize, hiddenSize]);
    const expected = [
      0.20053662,
      0.20053662,
      0.20053662,
      0.15482337,
      0.15482337,
      0.15482337,
      0.07484276,
      0.07484276,
      0.07484276,
    ];
    utils.checkValue(outputs.output.buffer, expected);
  });

  it('gru with 2 steps', async function() {
    const builder = nn.createModelBuilder();
    const steps = 2;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = builder.input(
        'input', {type: 'float32', dimensions: [steps, batchSize, inputSize]});
    const weight = builder.constant(
        {
          type: 'float32',
          dimensions: [numDirections, 3 * hiddenSize, inputSize],
        },
        new Float32Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = builder.constant(
        {
          type: 'float32',
          dimensions: [numDirections, 3 * hiddenSize, hiddenSize],
        },
        new Float32Array(numDirections * 3 * hiddenSize * hiddenSize)
            .fill(0.1));
    const initialHiddenState = builder.constant(
        {type: 'float32', dimensions: [numDirections, batchSize, hiddenSize]},
        new Float32Array(numDirections * batchSize * hiddenSize).fill(0));
    const bias = builder.constant(
        {type: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
        new Float32Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = builder.constant(
        {type: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
        new Float32Array(numDirections * 3 * hiddenSize).fill(0));
    const operands = builder.gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, initialHiddenState});
    const model = builder.createModel({output: operands[0]});
    const compiledModel = await model.compile();
    const inputs = {
      'input': {
        buffer: new Float32Array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
      },
    };
    const outputs = await compiledModel.compute(inputs);
    utils.checkShape(
        outputs.output.dimensions, [numDirections, batchSize, hiddenSize]);
    const expected = [
      0.22391089,
      0.22391089,
      0.22391089,
      0.22391089,
      0.22391089,
      0.1653014,
      0.1653014,
      0.1653014,
      0.1653014,
      0.1653014,
      0.0797327,
      0.0797327,
      0.0797327,
      0.0797327,
      0.0797327,
    ];
    utils.checkValue(outputs.output.buffer, expected);
  });

  it('gru without initialHiddenState', async function() {
    const builder = nn.createModelBuilder();
    const steps = 2;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = builder.input(
        'input', {type: 'float32', dimensions: [steps, batchSize, inputSize]});
    const weight = builder.constant(
        {
          type: 'float32',
          dimensions: [numDirections, 3 * hiddenSize, inputSize],
        },
        new Float32Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = builder.constant(
        {
          type: 'float32',
          dimensions: [numDirections, 3 * hiddenSize, hiddenSize],
        },
        new Float32Array(numDirections * 3 * hiddenSize * hiddenSize)
            .fill(0.1));
    const bias = builder.constant(
        {type: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
        new Float32Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = builder.constant(
        {type: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
        new Float32Array(numDirections * 3 * hiddenSize).fill(0));
    const operands = builder.gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias});
    const model = builder.createModel({output: operands[0]});
    const compiledModel = await model.compile();
    const inputs = {
      'input': {
        buffer: new Float32Array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
      },
    };
    const outputs = await compiledModel.compute(inputs);
    utils.checkShape(
        outputs.output.dimensions, [numDirections, batchSize, hiddenSize]);
    const expected = [
      0.22391089,
      0.22391089,
      0.22391089,
      0.22391089,
      0.22391089,
      0.1653014,
      0.1653014,
      0.1653014,
      0.1653014,
      0.1653014,
      0.0797327,
      0.0797327,
      0.0797327,
      0.0797327,
      0.0797327,
    ];
    utils.checkValue(outputs.output.buffer, expected);
  });
});
