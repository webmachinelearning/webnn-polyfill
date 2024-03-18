'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('test prelu converted from prelu test', async () => {
    // Converted test case (from: V1_2/prelu.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {dataType: 'float32', dimensions: [1, 2, 2, 3]});
    const inputData = new Float32Array([0, 0, 0, 1, 1, 1, -1, -1, -1, -2, -2, -2]);
    const alpha = builder.constant({dataType: 'float32', dimensions: [1, 1, 3]}, new Float32Array([0, 1, 2]));
    const expected = [0, 0, 0, 1, 1, 1, 0, -1, -2, 0, -2, -4];
    const output = builder.prelu(input, alpha);
    const graph = await builder.build({output});
    const outputs = {output: new Float32Array(utils.sizeOfShape([1, 2, 2, 3]))};
    const computeResult = await context.compute(graph, {'input': inputData}, outputs);
    utils.checkValue(computeResult.outputs.output, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test prelu converted from prelu_relaxed test', async () => {
    // Converted test case (from: V1_2/prelu.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {dataType: 'float32', dimensions: [1, 2, 2, 3]});
    const inputData = new Float32Array([0, 0, 0, 1, 1, 1, -1, -1, -1, -2, -2, -2]);
    const alpha = builder.constant({dataType: 'float32', dimensions: [1, 1, 3]}, new Float32Array([0, 1, 2]));
    const expected = [0, 0, 0, 1, 1, 1, 0, -1, -2, 0, -2, -4];
    const output = builder.prelu(input, alpha);
    const graph = await builder.build({output});
    const outputs = {output: new Float32Array(utils.sizeOfShape([1, 2, 2, 3]))};
    const computeResult = await context.compute(graph, {'input': inputData}, outputs);
    utils.checkValue(computeResult.outputs.output, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test prelu converted from prelu_weight_as_input test', async () => {
    // Converted test case (from: V1_2/prelu.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {dataType: 'float32', dimensions: [1, 2, 2, 3]});
    const inputData = new Float32Array([0, 0, 0, 1, 1, 1, -1, -1, -1, -2, -2, -2]);
    const alpha = builder.input('alpha', {dataType: 'float32', dimensions: [1, 1, 3]});
    const alphaData = new Float32Array([0, 1, 2]);
    const expected = [0, 0, 0, 1, 1, 1, 0, -1, -2, 0, -2, -4];
    const output = builder.prelu(input, alpha);
    const graph = await builder.build({output});
    const outputs = {output: new Float32Array(utils.sizeOfShape([1, 2, 2, 3]))};
    const computeResult = await context.compute(graph, {'input': inputData, 'alpha': alphaData}, outputs);
    utils.checkValue(computeResult.outputs.output, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test prelu converted from prelu_weight_as_input_relaxed test', async () => {
    // Converted test case (from: V1_2/prelu.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {dataType: 'float32', dimensions: [1, 2, 2, 3]});
    const inputData = new Float32Array([0, 0, 0, 1, 1, 1, -1, -1, -1, -2, -2, -2]);
    const alpha = builder.input('alpha', {dataType: 'float32', dimensions: [1, 1, 3]});
    const alphaData = new Float32Array([0, 1, 2]);
    const expected = [0, 0, 0, 1, 1, 1, 0, -1, -2, 0, -2, -4];
    const output = builder.prelu(input, alpha);
    const graph = await builder.build({output});
    const outputs = {output: new Float32Array(utils.sizeOfShape([1, 2, 2, 3]))};
    const computeResult = await context.compute(graph, {'input': inputData, 'alpha': alphaData}, outputs);
    utils.checkValue(computeResult.outputs.output, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
