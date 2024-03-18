'use strict';
import * as utils from '../utils.js';

describe('test gru', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('gruCell defaults', async () => {
    const builder = new MLGraphBuilder(context);
    const batchSize = 3;
    const inputSize = 2;
    const hiddenSize = 5;
    const input = builder.input(
        'input', {dataType: 'float32', dimensions: [batchSize, inputSize]});
    const weight = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize, inputSize]},
        new Float32Array(3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize, hiddenSize]},
        new Float32Array(3 * hiddenSize * hiddenSize).fill(0.1));
    const hiddenState = builder.constant(
        {dataType: 'float32', dimensions: [batchSize, hiddenSize]},
        new Float32Array(batchSize * hiddenSize).fill(0));
    const output = builder.gruCell(
        input, weight, recurrentWeight, hiddenState, hiddenSize);
    utils.checkDataType(output.dataType(), input.dataType());
    utils.checkShape(output.shape(), [batchSize, hiddenSize]);
    const graph = await builder.build({output});
    const inputs = {'input': new Float32Array([1, 2, 3, 4, 5, 6])};
    const outputs = {
      'output': new Float32Array(utils.sizeOfShape([batchSize, hiddenSize])),
    };
    const result = await context.compute(graph, inputs, outputs);
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
    utils.checkValue(result.outputs.output, expected);
  });

  it('gruCell with bias', async () => {
    const builder = new MLGraphBuilder(context);
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 3;
    const input = builder.input(
        'input', {dataType: 'float32', dimensions: [batchSize, inputSize]});
    const weight = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize, inputSize]},
        new Float32Array(3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize, hiddenSize]},
        new Float32Array(3 * hiddenSize * hiddenSize).fill(0.1));
    const hiddenState = builder.constant(
        {dataType: 'float32', dimensions: [batchSize, hiddenSize]},
        new Float32Array(batchSize * hiddenSize).fill(0));
    const bias = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize]},
        new Float32Array(3 * hiddenSize).fill(0.1));
    const output = builder.gruCell(
        input, weight, recurrentWeight, hiddenState, hiddenSize, {bias});
    utils.checkDataType(output.dataType(), input.dataType());
    utils.checkShape(output.shape(), [batchSize, hiddenSize]);
    const graph = await builder.build({output});
    const inputs = {'input': new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9])};
    const outputs = {
      'output': new Float32Array(utils.sizeOfShape([batchSize, hiddenSize])),
    };
    const result = await context.compute(graph, inputs, outputs);
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
    utils.checkValue(result.outputs.output, expected);
  });

  it('gruCell with recurrentBias', async () => {
    const builder = new MLGraphBuilder(context);
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 3;
    const input = builder.input(
        'input', {dataType: 'float32', dimensions: [batchSize, inputSize]});
    const weight = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize, inputSize]},
        new Float32Array(3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize, hiddenSize]},
        new Float32Array(3 * hiddenSize * hiddenSize).fill(0.1));
    const hiddenState = builder.constant(
        {dataType: 'float32', dimensions: [batchSize, hiddenSize]},
        new Float32Array(batchSize * hiddenSize).fill(0));
    const recurrentBias = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize]},
        new Float32Array(3 * hiddenSize).fill(1));
    const output = builder.gruCell(
        input, weight, recurrentWeight, hiddenState, hiddenSize,
        {recurrentBias});
    utils.checkDataType(output.dataType(), input.dataType());
    utils.checkShape(output.shape(), [batchSize, hiddenSize]);
    const graph = await builder.build({output});
    const inputs = {'input': new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9])};
    const outputs = {
      'output': new Float32Array(utils.sizeOfShape([batchSize, hiddenSize])),
    };
    const result = await context.compute(graph, inputs, outputs);
    const expected = [
      0.14985296,
      0.14985296,
      0.14985296,
      0.0746777,
      0.0746777,
      0.0746777,
      0.03221882,
      0.03221882,
      0.03221882,
    ];
    utils.checkValue(result.outputs.output, expected);
  });

  it('gruCell with explict resetAfter true', async () => {
    const builder = new MLGraphBuilder(context);
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 3;
    const input = builder.input(
        'input', {dataType: 'float32', dimensions: [batchSize, inputSize]});
    const weight = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize, inputSize]},
        new Float32Array(3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize, hiddenSize]},
        new Float32Array(3 * hiddenSize * hiddenSize).fill(0.1));
    const hiddenState = builder.constant(
        {dataType: 'float32', dimensions: [batchSize, hiddenSize]},
        new Float32Array(batchSize * hiddenSize).fill(2));
    const bias = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize]},
        new Float32Array(3 * hiddenSize).fill(0.1));
    const recurrentBias = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize]},
        new Float32Array(3 * hiddenSize).fill(1));
    const resetAfter = true;
    const output = builder.gruCell(
        input, weight, recurrentWeight, hiddenState, hiddenSize,
        {bias, recurrentBias, resetAfter});
    utils.checkDataType(output.dataType(), input.dataType());
    utils.checkShape(output.shape(), [batchSize, hiddenSize]);
    const graph = await builder.build({output});
    const inputs = {'input': new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9])};
    const outputs = {
      'output': new Float32Array(utils.sizeOfShape([batchSize, hiddenSize])),
    };
    const result = await context.compute(graph, inputs, outputs);
    const expected = [
      1.90645754,
      1.90645754,
      1.90645754,
      1.96068704,
      1.96068704,
      1.96068704,
      1.983688,
      1.983688,
      1.983688,
    ];
    utils.checkValue(result.outputs.output, expected);
  });

  it('gruCell with resetAfter false', async () => {
    const builder = new MLGraphBuilder(context);
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 3;
    const input = builder.input(
        'input', {dataType: 'float32', dimensions: [batchSize, inputSize]});
    const weight = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize, inputSize]},
        new Float32Array(3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize, hiddenSize]},
        new Float32Array(3 * hiddenSize * hiddenSize).fill(0.1));
    const hiddenState = builder.constant(
        {dataType: 'float32', dimensions: [batchSize, hiddenSize]},
        new Float32Array(batchSize * hiddenSize).fill(2));
    const bias = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize]},
        new Float32Array(3 * hiddenSize).fill(0.1));
    const recurrentBias = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize]},
        new Float32Array(3 * hiddenSize).fill(1));
    const resetAfter = false;
    const output = builder.gruCell(
        input, weight, recurrentWeight, hiddenState, hiddenSize,
        {bias, recurrentBias, resetAfter});
    utils.checkDataType(output.dataType(), input.dataType());
    utils.checkShape(output.shape(), [batchSize, hiddenSize]);
    const graph = await builder.build({output});
    const inputs = {'input': new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9])};
    const outputs = {
      'output': new Float32Array(utils.sizeOfShape([batchSize, hiddenSize])),
    };
    const result = await context.compute(graph, inputs, outputs);
    const expected = [
      1.90685618,
      1.90685618,
      1.90685618,
      1.96069813,
      1.96069813,
      1.96069813,
      1.98368835,
      1.98368835,
      1.98368835,
    ];
    utils.checkValue(result.outputs.output, expected);
  });

  it('gruCell with default zrn layout', async () => {
    const builder = new MLGraphBuilder(context);
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 3;
    const input = builder.input(
        'input', {dataType: 'float32', dimensions: [batchSize, inputSize]});
    const weight = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize, inputSize]},
        new Float32Array(3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize, hiddenSize]},
        new Float32Array(3 * hiddenSize * hiddenSize).fill(0.1));
    const hiddenState = builder.constant(
        {dataType: 'float32', dimensions: [batchSize, hiddenSize]},
        new Float32Array(batchSize * hiddenSize).fill(2));
    const bias = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize]},
        new Float32Array([
          1.9853785,
          2.2497437,
          0.6179927,
          0.3148022,
          -0.4366297,
          -0.9718124,
          -1.257099,
          -1.5698853,
          -0.39671835,
        ]),
    );
    const recurrentBias = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize]},
        new Float32Array(3 * hiddenSize).fill(1));
    const resetAfter = true;
    const output = builder.gruCell(
        input, weight, recurrentWeight, hiddenState, hiddenSize,
        {bias, recurrentBias, resetAfter});
    utils.checkDataType(output.dataType(), input.dataType());
    utils.checkShape(output.shape(), [batchSize, hiddenSize]);
    const graph = await builder.build({output});
    const inputs = {'input': new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9])};
    const outputs = {
      'output': new Float32Array(utils.sizeOfShape([batchSize, hiddenSize])),
    };
    const result = await context.compute(graph, inputs, outputs);
    const expected = [
      1.98016739,
      1.9812535,
      1.93765926,
      1.99351931,
      1.99475694,
      1.9759959,
      1.99746943,
      1.99804044,
      1.9902072,
    ];
    utils.checkValue(result.outputs.output, expected);
  });

  it('gruCell with explict zrn layout', async () => {
    const builder = new MLGraphBuilder(context);
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 3;
    const input = builder.input(
        'input', {dataType: 'float32', dimensions: [batchSize, inputSize]});
    const weight = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize, inputSize]},
        new Float32Array(3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize, hiddenSize]},
        new Float32Array(3 * hiddenSize * hiddenSize).fill(0.1));
    const hiddenState = builder.constant(
        {dataType: 'float32', dimensions: [batchSize, hiddenSize]},
        new Float32Array(batchSize * hiddenSize).fill(2));
    const bias = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize]},
        new Float32Array([
          1.9853785,
          2.2497437,
          0.6179927,
          0.3148022,
          -0.4366297,
          -0.9718124,
          -1.257099,
          -1.5698853,
          -0.39671835,
        ]),
    );
    const recurrentBias = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize]},
        new Float32Array(3 * hiddenSize).fill(1));
    const resetAfter = true;
    const layout = 'zrn';
    const output = builder.gruCell(
        input, weight, recurrentWeight, hiddenState, hiddenSize,
        {bias, recurrentBias, resetAfter, layout});
    utils.checkDataType(output.dataType(), input.dataType());
    utils.checkShape(output.shape(), [batchSize, hiddenSize]);
    const graph = await builder.build({output});
    const inputs = {'input': new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9])};
    const outputs = {
      'output': new Float32Array(utils.sizeOfShape([batchSize, hiddenSize])),
    };
    const result = await context.compute(graph, inputs, outputs);
    const expected = [
      1.98016739,
      1.9812535,
      1.93765926,
      1.99351931,
      1.99475694,
      1.9759959,
      1.99746943,
      1.99804044,
      1.9902072,
    ];
    utils.checkValue(result.outputs.output, expected);
  });

  it('gruCell with rzn layout', async () => {
    const builder = new MLGraphBuilder(context);
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 3;
    const input = builder.input(
        'input', {dataType: 'float32', dimensions: [batchSize, inputSize]});
    const weight = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize, inputSize]},
        new Float32Array(3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize, hiddenSize]},
        new Float32Array(3 * hiddenSize * hiddenSize).fill(0.1));
    const hiddenState = builder.constant(
        {dataType: 'float32', dimensions: [batchSize, hiddenSize]},
        new Float32Array(batchSize * hiddenSize).fill(2));
    const bias = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize]},
        new Float32Array([
          0.3148022,
          -0.4366297,
          -0.9718124,
          1.9853785,
          2.2497437,
          0.6179927,
          -1.257099,
          -1.5698853,
          -0.39671835,
        ]),
    );
    const recurrentBias = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize]},
        new Float32Array(3 * hiddenSize).fill(1));
    const resetAfter = true;
    const layout = 'rzn';
    const output = builder.gruCell(
        input, weight, recurrentWeight, hiddenState, hiddenSize,
        {bias, recurrentBias, resetAfter, layout});
    utils.checkDataType(output.dataType(), input.dataType());
    utils.checkShape(output.shape(), [batchSize, hiddenSize]);
    const graph = await builder.build({output});
    const inputs = {'input': new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9])};
    const outputs = {
      'output': new Float32Array(utils.sizeOfShape([batchSize, hiddenSize])),
    };
    const result = await context.compute(graph, inputs, outputs);
    const expected = [
      1.98016739,
      1.9812535,
      1.93765926,
      1.99351931,
      1.99475694,
      1.9759959,
      1.99746943,
      1.99804044,
      1.9902072,
    ];
    utils.checkValue(result.outputs.output, expected);
  });

  it('gruCell with [tanh, sigmoid] activations', async () => {
    const builder = new MLGraphBuilder(context);
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 3;
    const input = builder.input(
        'input', {dataType: 'float32', dimensions: [batchSize, inputSize]});
    const weight = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize, inputSize]},
        new Float32Array(3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize, hiddenSize]},
        new Float32Array(3 * hiddenSize * hiddenSize).fill(0.1));
    const hiddenState = builder.constant(
        {dataType: 'float32', dimensions: [batchSize, hiddenSize]},
        new Float32Array(batchSize * hiddenSize).fill(2));
    const bias = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize]},
        new Float32Array([
          1.9853785,
          2.2497437,
          0.6179927,
          0.3148022,
          -0.4366297,
          -0.9718124,
          -1.257099,
          -1.5698853,
          -0.39671835,
        ]),
    );
    const recurrentBias = builder.constant(
        {dataType: 'float32', dimensions: [3 * hiddenSize]},
        new Float32Array(3 * hiddenSize).fill(1));
    const resetAfter = true;
    const output = builder.gruCell(
        input, weight, recurrentWeight, hiddenState, hiddenSize,
        {
          bias,
          recurrentBias,
          resetAfter,
          activations: [builder.tanh(), builder.sigmoid()],
        });
    utils.checkDataType(output.dataType(), input.dataType());
    utils.checkShape(output.shape(), [batchSize, hiddenSize]);
    const graph = await builder.build({output});
    const inputs = {'input': new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9])};
    const outputs = {
      'output': new Float32Array(utils.sizeOfShape([batchSize, hiddenSize])),
    };
    const result = await context.compute(graph, inputs, outputs);
    const expected = [
      1.99940538,
      1.99962664,
      1.99164689,
      1.99991298,
      1.99994671,
      1.99874425,
      1.99998665,
      1.99999189,
      1.99979985,
    ];
    utils.checkValue(result.outputs.output, expected);
  });

  it('gru with 1 step', async () => {
    const builder = new MLGraphBuilder(context);
    const steps = 1;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 3;
    const input = builder.input(
        'input',
        {dataType: 'float32', dimensions: [steps, batchSize, inputSize]});
    const weight = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, 3 * hiddenSize, inputSize],
        },
        new Float32Array(numDirections * 3 * hiddenSize * inputSize)
            .fill(0.1));
    const recurrentWeight = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, 3 * hiddenSize, hiddenSize],
        },
        new Float32Array(numDirections * 3 * hiddenSize * hiddenSize)
            .fill(0.1));
    const bias = builder.constant(
        {dataType: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
        new Float32Array([
          0.3148022,
          -0.4366297,
          -0.9718124,
          1.9853785,
          2.2497437,
          0.6179927,
          -1.257099,
          -1.5698853,
          -0.39671835,
        ]),
    );
    const recurrentBias = builder.constant(
        {dataType: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
        new Float32Array(numDirections * 3 * hiddenSize).fill(1));
    const initialHiddenState = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, batchSize, hiddenSize],
        },
        new Float32Array(numDirections * batchSize * hiddenSize).fill(2));
    const resetAfter = true;
    const layout = 'rzn';
    const operands = builder.gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, initialHiddenState, resetAfter, layout});
    utils.checkDataType(operands[0].dataType(), input.dataType());
    utils.checkShape(
        operands[0].shape(), [numDirections, batchSize, hiddenSize]);
    const graph = await builder.build({output: operands[0]});
    const inputs = {'input': new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9])};
    const outputs = {
      'output': new Float32Array(
          utils.sizeOfShape([numDirections, batchSize, hiddenSize])),
    };
    const result = await context.compute(graph, inputs, outputs);
    const expected = [
      1.98016739,
      1.9812535,
      1.93765926,
      1.99351931,
      1.99475694,
      1.9759959,
      1.99746943,
      1.99804044,
      1.9902072,
    ];
    utils.checkValue(result.outputs.output, expected);
  });

  it('gru with 2 steps', async () => {
    const builder = new MLGraphBuilder(context);
    const steps = 2;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = builder.input(
        'input',
        {dataType: 'float32', dimensions: [steps, batchSize, inputSize]});
    const weight = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, 3 * hiddenSize, inputSize],
        },
        new Float32Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, 3 * hiddenSize, hiddenSize],
        },
        new Float32Array(numDirections * 3 * hiddenSize * hiddenSize)
            .fill(0.1));
    const initialHiddenState = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, batchSize, hiddenSize],
        },
        new Float32Array(numDirections * batchSize * hiddenSize).fill(0));
    const bias = builder.constant(
        {dataType: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
        new Float32Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = builder.constant(
        {dataType: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
        new Float32Array(numDirections * 3 * hiddenSize).fill(0));
    const operands = builder.gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, initialHiddenState});
    utils.checkDataType(operands[0].dataType(), input.dataType());
    utils.checkShape(
        operands[0].shape(), [numDirections, batchSize, hiddenSize]);
    const graph = await builder.build({output: operands[0]});
    const inputs = {
      'input': new Float32Array(
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
    };
    const outputs = {
      'output': new Float32Array(
          utils.sizeOfShape([numDirections, batchSize, hiddenSize])),
    };
    const result = await context.compute(graph, inputs, outputs);
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
    utils.checkValue(result.outputs.output, expected);
  });

  it('gru with explict returnSequence false', async () => {
    const builder = new MLGraphBuilder(context);
    const steps = 2;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = builder.input(
        'input',
        {dataType: 'float32', dimensions: [steps, batchSize, inputSize]});
    const weight = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, 3 * hiddenSize, inputSize],
        },
        new Float32Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, 3 * hiddenSize, hiddenSize],
        },
        new Float32Array(numDirections * 3 * hiddenSize * hiddenSize)
            .fill(0.1));
    const initialHiddenState = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, batchSize, hiddenSize],
        },
        new Float32Array(numDirections * batchSize * hiddenSize).fill(0));
    const bias = builder.constant(
        {dataType: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
        new Float32Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = builder.constant(
        {dataType: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
        new Float32Array(numDirections * 3 * hiddenSize).fill(0));
    const returnSequence = false;
    const operands = builder.gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, initialHiddenState, returnSequence});
    utils.checkDataType(operands[0].dataType(), input.dataType());
    utils.checkShape(
        operands[0].shape(), [numDirections, batchSize, hiddenSize]);
    const graph = await builder.build({output: operands[0]});
    const inputs = {
      'input': new Float32Array(
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
    };
    const outputs = {
      'output': new Float32Array(
          utils.sizeOfShape([numDirections, batchSize, hiddenSize])),
    };
    const result = await context.compute(graph, inputs, outputs);
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
    utils.checkValue(result.outputs.output, expected);
  });

  it('gru with returnSequence true', async () => {
    const builder = new MLGraphBuilder(context);
    const steps = 2;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = builder.input(
        'input',
        {dataType: 'float32', dimensions: [steps, batchSize, inputSize]});
    const weight = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, 3 * hiddenSize, inputSize],
        },
        new Float32Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, 3 * hiddenSize, hiddenSize],
        },
        new Float32Array(numDirections * 3 * hiddenSize * hiddenSize)
            .fill(0.1));
    const initialHiddenState = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, batchSize, hiddenSize],
        },
        new Float32Array(numDirections * batchSize * hiddenSize).fill(0));
    const bias = builder.constant(
        {dataType: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
        new Float32Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = builder.constant(
        {dataType: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
        new Float32Array(numDirections * 3 * hiddenSize).fill(0));
    const returnSequence = true;
    const operands = builder.gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, initialHiddenState, returnSequence});
    utils.checkDataType(operands[0].dataType(), input.dataType());
    utils.checkShape(
        operands[0].shape(), [numDirections, batchSize, hiddenSize]);
    utils.checkDataType(operands[1].dataType(), input.dataType());
    utils.checkShape(
        operands[1].shape(), [steps, numDirections, batchSize, hiddenSize]);
    const graph = await builder.build(
        {output0: operands[0], output1: operands[1]});
    const inputs = {
      'input': new Float32Array(
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
    };
    const outputs = {
      'output0': new Float32Array(
          utils.sizeOfShape([numDirections, batchSize, hiddenSize])),
      'output1': new Float32Array(
          utils.sizeOfShape([steps, numDirections, batchSize, hiddenSize])),
    };
    const result = await context.compute(graph, inputs, outputs);
    const expected = [
      [
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
      ],
      [
        0.20053661,
        0.20053661,
        0.20053661,
        0.20053661,
        0.20053661,
        0.15482338,
        0.15482338,
        0.15482338,
        0.15482338,
        0.15482338,
        0.07484276,
        0.07484276,
        0.07484276,
        0.07484276,
        0.07484276,
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
      ],
    ];
    for (let i = 0; i < expected.length; ++i) {
      utils.checkValue(result.outputs[`output${i}`], expected[i]);
    }
  });

  it('gru with explict forward direction', async () => {
    const builder = new MLGraphBuilder(context);
    const steps = 2;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = builder.input(
        'input',
        {dataType: 'float32', dimensions: [steps, batchSize, inputSize]});
    const weight = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, 3 * hiddenSize, inputSize],
        },
        new Float32Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, 3 * hiddenSize, hiddenSize],
        },
        new Float32Array(numDirections * 3 * hiddenSize * hiddenSize)
            .fill(0.1));
    const initialHiddenState = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, batchSize, hiddenSize],
        },
        new Float32Array(numDirections * batchSize * hiddenSize).fill(0));
    const bias = builder.constant(
        {dataType: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
        new Float32Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = builder.constant(
        {dataType: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
        new Float32Array(numDirections * 3 * hiddenSize).fill(0));
    const direction = 'forward';
    const operands = builder.gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, initialHiddenState, direction});
    utils.checkDataType(operands[0].dataType(), input.dataType());
    utils.checkShape(
        operands[0].shape(), [numDirections, batchSize, hiddenSize]);
    const graph = await builder.build({output: operands[0]});
    const inputs = {
      'input': new Float32Array(
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
    };
    const outputs = {
      'output': new Float32Array(
          utils.sizeOfShape([numDirections, batchSize, hiddenSize])),
    };
    const result = await context.compute(graph, inputs, outputs);
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
    utils.checkValue(result.outputs.output, expected);
  });

  it('gru with backward direction', async () => {
    const builder = new MLGraphBuilder(context);
    const steps = 2;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = builder.input(
        'input',
        {dataType: 'float32', dimensions: [steps, batchSize, inputSize]});
    const weight = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, 3 * hiddenSize, inputSize],
        },
        new Float32Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, 3 * hiddenSize, hiddenSize],
        },
        new Float32Array(numDirections * 3 * hiddenSize * hiddenSize)
            .fill(0.1));
    const initialHiddenState = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, batchSize, hiddenSize],
        },
        new Float32Array(numDirections * batchSize * hiddenSize).fill(0));
    const bias = builder.constant(
        {dataType: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
        new Float32Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = builder.constant(
        {dataType: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
        new Float32Array(numDirections * 3 * hiddenSize).fill(0));
    const direction = 'backward';
    const operands = builder.gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, initialHiddenState, direction});
    utils.checkDataType(operands[0].dataType(), input.dataType());
    utils.checkShape(
        operands[0].shape(), [numDirections, batchSize, hiddenSize]);
    const graph = await builder.build({output: operands[0]});
    const inputs = {
      'input': new Float32Array(
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
    };
    const outputs = {
      'output': new Float32Array(
          utils.sizeOfShape([numDirections, batchSize, hiddenSize])),
    };
    const result = await context.compute(graph, inputs, outputs);
    const expected = [
      0.22227009,
      0.22227009,
      0.22227009,
      0.22227009,
      0.22227009,
      0.16524932,
      0.16524932,
      0.16524932,
      0.16524932,
      0.16524932,
      0.07972924,
      0.07972924,
      0.07972924,
      0.07972924,
      0.07972924,
    ];
    utils.checkValue(result.outputs.output, expected);
  });

  it('gru with both direction', async () => {
    const builder = new MLGraphBuilder(context);
    const steps = 2;
    const numDirections = 2;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = builder.input(
        'input',
        {dataType: 'float32', dimensions: [steps, batchSize, inputSize]});
    const weight = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, 3 * hiddenSize, inputSize],
        },
        new Float32Array(numDirections * 3 * hiddenSize * inputSize).fill(0.1));
    const recurrentWeight = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, 3 * hiddenSize, hiddenSize],
        },
        new Float32Array(numDirections * 3 * hiddenSize * hiddenSize)
            .fill(0.1));
    const initialHiddenState = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, batchSize, hiddenSize],
        },
        new Float32Array(numDirections * batchSize * hiddenSize).fill(0));
    const bias = builder.constant(
        {dataType: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
        new Float32Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = builder.constant(
        {dataType: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
        new Float32Array(numDirections * 3 * hiddenSize).fill(0));
    const direction = 'both';
    const operands = builder.gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias, initialHiddenState, direction});
    utils.checkDataType(operands[0].dataType(), input.dataType());
    utils.checkShape(
        operands[0].shape(), [numDirections, batchSize, hiddenSize]);
    const graph = await builder.build({output: operands[0]});
    const inputs = {
      'input': new Float32Array(
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
    };
    const outputs = {
      'output': new Float32Array(
          utils.sizeOfShape([numDirections, batchSize, hiddenSize])),
    };
    const result = await context.compute(graph, inputs, outputs);
    const expected = [
      0.2239109,
      0.2239109,
      0.2239109,
      0.2239109,
      0.2239109,
      0.16530138,
      0.16530138,
      0.16530138,
      0.16530138,
      0.16530138,
      0.07973271,
      0.07973271,
      0.07973271,
      0.07973271,
      0.07973271,
      0.22227009,
      0.22227009,
      0.22227009,
      0.22227009,
      0.22227009,
      0.16524932,
      0.16524932,
      0.16524932,
      0.16524932,
      0.16524932,
      0.07972924,
      0.07972924,
      0.07972924,
      0.07972924,
      0.07972924,
    ];
    utils.checkValue(result.outputs.output, expected);
  });

  it('gru without initialHiddenState', async () => {
    const builder = new MLGraphBuilder(context);
    const steps = 2;
    const numDirections = 1;
    const batchSize = 3;
    const inputSize = 3;
    const hiddenSize = 5;
    const input = builder.input(
        'input',
        {dataType: 'float32', dimensions: [steps, batchSize, inputSize]});
    const weight = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, 3 * hiddenSize, inputSize],
        },
        new Float32Array(numDirections * 3 * hiddenSize * inputSize)
            .fill(0.1));
    const recurrentWeight = builder.constant(
        {
          dataType: 'float32',
          dimensions: [numDirections, 3 * hiddenSize, hiddenSize],
        },
        new Float32Array(numDirections * 3 * hiddenSize * hiddenSize)
            .fill(0.1));
    const bias = builder.constant(
        {dataType: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
        new Float32Array(numDirections * 3 * hiddenSize).fill(0.1));
    const recurrentBias = builder.constant(
        {dataType: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
        new Float32Array(numDirections * 3 * hiddenSize).fill(0));
    const operands = builder.gru(
        input, weight, recurrentWeight, steps, hiddenSize,
        {bias, recurrentBias});
    utils.checkDataType(operands[0].dataType(), input.dataType());
    utils.checkShape(
        operands[0].shape(), [numDirections, batchSize, hiddenSize]);
    const graph = await builder.build({output: operands[0]});
    const inputs = {
      'input': new Float32Array(
          [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
    };
    const outputs = {
      'output': new Float32Array(
          utils.sizeOfShape([numDirections, batchSize, hiddenSize])),
    };
    const result = await context.compute(graph, inputs, outputs);
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
    utils.checkValue(result.outputs.output, expected);
  });
});
