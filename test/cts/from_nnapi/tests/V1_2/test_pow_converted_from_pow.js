'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('test pow converted from pow test', async () => {
    // Converted test case (from: V1_2/pow.mod.py)
    const builder = new MLGraphBuilder(context);
    const base = builder.input('base', {type: 'float32', dimensions: [2, 1]});
    const baseData = new Float32Array([2.0, 3.0]);
    const exponent = builder.input('exponent', {type: 'float32', dimensions: [1]});
    const exponentData = new Float32Array([2.0]);
    const expected = [4.0, 9.0];
    const output = builder.pow(base, exponent);
    const graph = await builder.build({output});
    const outputs = {output: new Float32Array(utils.sizeOfShape([2, 1]))};
    await context.compute(graph, {'base': baseData, 'exponent': exponentData}, outputs);
    utils.checkValue(outputs.output, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test pow converted from pow_relaxed test', async () => {
    // Converted test case (from: V1_2/pow.mod.py)
    const builder = new MLGraphBuilder(context);
    const base = builder.input('base', {type: 'float32', dimensions: [2, 1]});
    const baseData = new Float32Array([2.0, 3.0]);
    const exponent = builder.input('exponent', {type: 'float32', dimensions: [1]});
    const exponentData = new Float32Array([2.0]);
    const expected = [4.0, 9.0];
    const output = builder.pow(base, exponent);
    const graph = await builder.build({output});
    const outputs = {output: new Float32Array(utils.sizeOfShape([2, 1]))};
    await context.compute(graph, {'base': baseData, 'exponent': exponentData}, outputs);
    utils.checkValue(outputs.output, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test pow converted from pow_2 test', async () => {
    // Converted test case (from: V1_2/pow.mod.py)
    const builder = new MLGraphBuilder(context);
    const base = builder.input('base', {type: 'float32', dimensions: [2, 1]});
    const baseData = new Float32Array([2.0, 3.0]);
    const exponent1 = builder.input('exponent1', {type: 'float32', dimensions: [1, 2]});
    const exponent1Data = new Float32Array([2.0, 3.0]);
    const expected = [4.0, 8.0, 9.0, 27.0];
    const output1 = builder.pow(base, exponent1);
    const graph = await builder.build({output1});
    const outputs = {output1: new Float32Array(utils.sizeOfShape([2, 2]))};
    await context.compute(graph, {'base': baseData, 'exponent1': exponent1Data}, outputs);
    utils.checkValue(outputs.output1, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test pow converted from pow_relaxed_2 test', async () => {
    // Converted test case (from: V1_2/pow.mod.py)
    const builder = new MLGraphBuilder(context);
    const base = builder.input('base', {type: 'float32', dimensions: [2, 1]});
    const baseData = new Float32Array([2.0, 3.0]);
    const exponent1 = builder.input('exponent1', {type: 'float32', dimensions: [1, 2]});
    const exponent1Data = new Float32Array([2.0, 3.0]);
    const expected = [4.0, 8.0, 9.0, 27.0];
    const output1 = builder.pow(base, exponent1);
    const graph = await builder.build({output1});
    const outputs = {output1: new Float32Array(utils.sizeOfShape([2, 2]))};
    await context.compute(graph, {'base': baseData, 'exponent1': exponent1Data}, outputs);
    utils.checkValue(outputs.output1, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test pow converted from pow_3 test', async () => {
    // Converted test case (from: V1_2/pow.mod.py)
    const builder = new MLGraphBuilder(context);
    const base = builder.input('base', {type: 'float32', dimensions: [2, 1]});
    const baseData = new Float32Array([2.0, 3.0]);
    const exponent2 = builder.input('exponent2', {type: 'float32', dimensions: [3, 1, 2]});
    const exponent2Data = new Float32Array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0]);
    const expected = [1.0, 1.4142135623730951, 1.0, 1.7320508075688772, 2.0, 4.0, 3.0, 9.0, 8.0, 16.0, 27.0, 81.0];
    const output2 = builder.pow(base, exponent2);
    const graph = await builder.build({output2});
    const outputs = {output2: new Float32Array(utils.sizeOfShape([3, 2, 2]))};
    await context.compute(graph, {'base': baseData, 'exponent2': exponent2Data}, outputs);
    utils.checkValue(outputs.output2, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test pow converted from pow_relaxed_3 test', async () => {
    // Converted test case (from: V1_2/pow.mod.py)
    const builder = new MLGraphBuilder(context);
    const base = builder.input('base', {type: 'float32', dimensions: [2, 1]});
    const baseData = new Float32Array([2.0, 3.0]);
    const exponent2 = builder.input('exponent2', {type: 'float32', dimensions: [3, 1, 2]});
    const exponent2Data = new Float32Array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0]);
    const expected = [1.0, 1.4142135623730951, 1.0, 1.7320508075688772, 2.0, 4.0, 3.0, 9.0, 8.0, 16.0, 27.0, 81.0];
    const output2 = builder.pow(base, exponent2);
    const graph = await builder.build({output2});
    const outputs = {output2: new Float32Array(utils.sizeOfShape([3, 2, 2]))};
    await context.compute(graph, {'base': baseData, 'exponent2': exponent2Data}, outputs);
    utils.checkValue(outputs.output2, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
