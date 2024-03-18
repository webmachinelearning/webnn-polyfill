'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('test max converted from maximum_simple test', async () => {
    // Converted test case (from: V1_2/maximum.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {dataType: 'float32', dimensions: [3, 1, 2]});
    const input0Data = new Float32Array([1.0, 0.0, -1.0, 11.0, -2.0, -1.44]);
    const input1 = builder.input('input1', {dataType: 'float32', dimensions: [3, 1, 2]});
    const input1Data = new Float32Array([-1.0, 0.0, 1.0, 12.0, -3.0, -1.43]);
    const expected = [1.0, 0.0, 1.0, 12.0, -2.0, -1.43];
    const output0 = builder.max(input0, input1);
    const graph = await builder.build({output0});
    const outputs = {output0: new Float32Array(utils.sizeOfShape([3, 1, 2]))};
    const computeResult = await context.compute(graph, {'input0': input0Data, 'input1': input1Data}, outputs);
    utils.checkValue(computeResult.outputs.output0, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test max converted from maximum_simple_relaxed test', async () => {
    // Converted test case (from: V1_2/maximum.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {dataType: 'float32', dimensions: [3, 1, 2]});
    const input0Data = new Float32Array([1.0, 0.0, -1.0, 11.0, -2.0, -1.44]);
    const input1 = builder.input('input1', {dataType: 'float32', dimensions: [3, 1, 2]});
    const input1Data = new Float32Array([-1.0, 0.0, 1.0, 12.0, -3.0, -1.43]);
    const expected = [1.0, 0.0, 1.0, 12.0, -2.0, -1.43];
    const output0 = builder.max(input0, input1);
    const graph = await builder.build({output0});
    const outputs = {output0: new Float32Array(utils.sizeOfShape([3, 1, 2]))};
    const computeResult = await context.compute(graph, {'input0': input0Data, 'input1': input1Data}, outputs);
    utils.checkValue(computeResult.outputs.output0, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test max converted from maximum_simple_int32 test', async () => {
    // Converted test case (from: V1_2/maximum.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {dataType: 'int32', dimensions: [3, 1, 2]});
    const input0Data = new Int32Array([1, 0, -1, 11, -2, -1]);
    const input1 = builder.input('input1', {dataType: 'int32', dimensions: [3, 1, 2]});
    const input1Data = new Int32Array([-1, 0, 1, 12, -3, -1]);
    const expected = [1, 0, 1, 12, -2, -1];
    const output0 = builder.max(input0, input1);
    const graph = await builder.build({output0});
    const outputs = {output0: new Int32Array(utils.sizeOfShape([3, 1, 2]))};
    const computeResult = await context.compute(graph, {'input0': input0Data, 'input1': input1Data}, outputs);
    utils.checkValue(computeResult.outputs.output0, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test max converted from maximum_broadcast test', async () => {
    // Converted test case (from: V1_2/maximum.mod.py)
    const builder = new MLGraphBuilder(context);
    const input01 = builder.input('input01', {dataType: 'float32', dimensions: [3, 1, 2]});
    const input01Data = new Float32Array([1.0, 0.0, -1.0, -2.0, -1.44, 11.0]);
    const input11 = builder.input('input11', {dataType: 'float32', dimensions: [2]});
    const input11Data = new Float32Array([0.5, 2.0]);
    const expected = [1.0, 2.0, 0.5, 2.0, 0.5, 11.0];
    const output01 = builder.max(input01, input11);
    const graph = await builder.build({output01});
    const outputs = {output01: new Float32Array(utils.sizeOfShape([3, 1, 2]))};
    const computeResult = await context.compute(graph, {'input01': input01Data, 'input11': input11Data}, outputs);
    utils.checkValue(computeResult.outputs.output01, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test max converted from maximum_broadcast_relaxed test', async () => {
    // Converted test case (from: V1_2/maximum.mod.py)
    const builder = new MLGraphBuilder(context);
    const input01 = builder.input('input01', {dataType: 'float32', dimensions: [3, 1, 2]});
    const input01Data = new Float32Array([1.0, 0.0, -1.0, -2.0, -1.44, 11.0]);
    const input11 = builder.input('input11', {dataType: 'float32', dimensions: [2]});
    const input11Data = new Float32Array([0.5, 2.0]);
    const expected = [1.0, 2.0, 0.5, 2.0, 0.5, 11.0];
    const output01 = builder.max(input01, input11);
    const graph = await builder.build({output01});
    const outputs = {output01: new Float32Array(utils.sizeOfShape([3, 1, 2]))};
    const computeResult = await context.compute(graph, {'input01': input01Data, 'input11': input11Data}, outputs);
    utils.checkValue(computeResult.outputs.output01, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test max converted from maximum_broadcast_int32 test', async () => {
    // Converted test case (from: V1_2/maximum.mod.py)
    const builder = new MLGraphBuilder(context);
    const input01 = builder.input('input01', {dataType: 'int32', dimensions: [3, 1, 2]});
    const input01Data = new Int32Array([1, 0, -1, -2, -1, 11]);
    const input11 = builder.input('input11', {dataType: 'int32', dimensions: [2]});
    const input11Data = new Int32Array([0, 2]);
    const expected = [1, 2, 0, 2, 0, 11];
    const output01 = builder.max(input01, input11);
    const graph = await builder.build({output01});
    const outputs = {output01: new Int32Array(utils.sizeOfShape([3, 1, 2]))};
    const computeResult = await context.compute(graph, {'input01': input01Data, 'input11': input11Data}, outputs);
    utils.checkValue(computeResult.outputs.output01, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
