'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('test sub + clamp converted from sub_v1_2_broadcast_none test', async () => {
    // Converted test case (from: V1_2/sub_v1_2_broadcast.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {dataType: 'float32', dimensions: [1, 2]});
    const input0Data = new Float32Array([10, 20]);
    const input1 = builder.input('input1', {dataType: 'float32', dimensions: [2, 2]});
    const input1Data = new Float32Array([0.1, 0.2, 0.3, 0.4]);
    const expected = [9.9, 19.8, 9.7, 19.6];
    const interOut0 = builder.sub(input0, input1);
    const output0 = builder.clamp(interOut0);
    const graph = await builder.build({output0});
    const outputs = {output0: new Float32Array(utils.sizeOfShape([2, 2]))};
    const computeResult = await context.compute(graph, {'input0': input0Data, 'input1': input1Data}, outputs);
    utils.checkValue(computeResult.outputs.output0, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test sub + clamp converted from sub_v1_2_broadcast_relu test', async () => {
    // Converted test case (from: V1_2/sub_v1_2_broadcast.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {dataType: 'float32', dimensions: [1, 2]});
    const input0Data = new Float32Array([10, 20]);
    const input1 = builder.input('input1', {dataType: 'float32', dimensions: [2, 2]});
    const input1Data = new Float32Array([0.1, 0.2, 0.3, 0.4]);
    const expected = [9.9, 19.8, 9.7, 19.6];
    const interOut0 = builder.sub(input0, input1);
    const output0 = builder.relu(interOut0);
    const graph = await builder.build({output0});
    const outputs = {output0: new Float32Array(utils.sizeOfShape([2, 2]))};
    const computeResult = await context.compute(graph, {'input0': input0Data, 'input1': input1Data}, outputs);
    utils.checkValue(computeResult.outputs.output0, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test sub + clamp converted from sub_v1_2_broadcast_relu1 test', async () => {
    // Converted test case (from: V1_2/sub_v1_2_broadcast.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {dataType: 'float32', dimensions: [1, 2]});
    const input0Data = new Float32Array([10, 20]);
    const input1 = builder.input('input1', {dataType: 'float32', dimensions: [2, 2]});
    const input1Data = new Float32Array([0.1, 0.2, 0.3, 0.4]);
    const expected = [1.0, 1.0, 1.0, 1.0];
    const interOut0 = builder.sub(input0, input1);
    const output0 = builder.clamp(interOut0, {minValue: -1, maxValue: 1});
    const graph = await builder.build({output0});
    const outputs = {output0: new Float32Array(utils.sizeOfShape([2, 2]))};
    const computeResult = await context.compute(graph, {'input0': input0Data, 'input1': input1Data}, outputs);
    utils.checkValue(computeResult.outputs.output0, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test sub + clamp converted from sub_v1_2_broadcast_relu6 test', async () => {
    // Converted test case (from: V1_2/sub_v1_2_broadcast.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {dataType: 'float32', dimensions: [1, 2]});
    const input0Data = new Float32Array([10, 20]);
    const input1 = builder.input('input1', {dataType: 'float32', dimensions: [2, 2]});
    const input1Data = new Float32Array([0.1, 0.2, 0.3, 0.4]);
    const expected = [6.0, 6.0, 6.0, 6.0];
    const interOut0 = builder.sub(input0, input1);
    const output0 = builder.clamp(interOut0, {minValue: 0, maxValue: 6});
    const graph = await builder.build({output0});
    const outputs = {output0: new Float32Array(utils.sizeOfShape([2, 2]))};
    const computeResult = await context.compute(graph, {'input0': input0Data, 'input1': input1Data}, outputs);
    utils.checkValue(computeResult.outputs.output0, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
