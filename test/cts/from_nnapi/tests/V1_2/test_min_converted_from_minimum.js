'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test min converted from minimum_simple test', async function() {
    // Converted test case (from: V1_2/minimum.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {type: 'float32', dimensions: [3, 1, 2]});
    const input0Data = new Float32Array([1.0, 0.0, -1.0, 11.0, -2.0, -1.44]);
    const input1 = builder.input('input1', {type: 'float32', dimensions: [3, 1, 2]});
    const input1Data = new Float32Array([-1.0, 0.0, 1.0, 12.0, -3.0, -1.43]);
    const expected = [-1.0, 0.0, -1.0, 11.0, -3.0, -1.44];
    const output0 = builder.min(input0, input1);
    const graph = await builder.build({output0});
    const outputs = await graph.compute({'input0': {data: input0Data}, 'input1': {data: input1Data}});
    utils.checkValue(outputs.output0.data, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test min converted from minimum_simple_relaxed test', async function() {
    // Converted test case (from: V1_2/minimum.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {type: 'float32', dimensions: [3, 1, 2]});
    const input0Data = new Float32Array([1.0, 0.0, -1.0, 11.0, -2.0, -1.44]);
    const input1 = builder.input('input1', {type: 'float32', dimensions: [3, 1, 2]});
    const input1Data = new Float32Array([-1.0, 0.0, 1.0, 12.0, -3.0, -1.43]);
    const expected = [-1.0, 0.0, -1.0, 11.0, -3.0, -1.44];
    const output0 = builder.min(input0, input1);
    const graph = await builder.build({output0});
    const outputs = await graph.compute({'input0': {data: input0Data}, 'input1': {data: input1Data}});
    utils.checkValue(outputs.output0.data, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test min converted from minimum_simple_int32 test', async function() {
    // Converted test case (from: V1_2/minimum.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {type: 'int32', dimensions: [3, 1, 2]});
    const input0Data = new Int32Array([1, 0, -1, 11, -2, -1]);
    const input1 = builder.input('input1', {type: 'int32', dimensions: [3, 1, 2]});
    const input1Data = new Int32Array([-1, 0, 1, 12, -3, -1]);
    const expected = [-1, 0, -1, 11, -3, -1];
    const output0 = builder.min(input0, input1);
    const graph = await builder.build({output0});
    const outputs = await graph.compute({'input0': {data: input0Data}, 'input1': {data: input1Data}});
    utils.checkValue(outputs.output0.data, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test min converted from minimum_broadcast test', async function() {
    // Converted test case (from: V1_2/minimum.mod.py)
    const builder = new MLGraphBuilder(context);
    const input01 = builder.input('input01', {type: 'float32', dimensions: [3, 1, 2]});
    const input01Data = new Float32Array([1.0, 0.0, -1.0, -2.0, -1.44, 11.0]);
    const input11 = builder.input('input11', {type: 'float32', dimensions: [2]});
    const input11Data = new Float32Array([0.5, 2.0]);
    const expected = [0.5, 0.0, -1.0, -2.0, -1.44, 2.0];
    const output01 = builder.min(input01, input11);
    const graph = await builder.build({output01});
    const outputs = await graph.compute({'input01': {data: input01Data}, 'input11': {data: input11Data}});
    utils.checkValue(outputs.output01.data, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test min converted from minimum_broadcast_relaxed test', async function() {
    // Converted test case (from: V1_2/minimum.mod.py)
    const builder = new MLGraphBuilder(context);
    const input01 = builder.input('input01', {type: 'float32', dimensions: [3, 1, 2]});
    const input01Data = new Float32Array([1.0, 0.0, -1.0, -2.0, -1.44, 11.0]);
    const input11 = builder.input('input11', {type: 'float32', dimensions: [2]});
    const input11Data = new Float32Array([0.5, 2.0]);
    const expected = [0.5, 0.0, -1.0, -2.0, -1.44, 2.0];
    const output01 = builder.min(input01, input11);
    const graph = await builder.build({output01});
    const outputs = await graph.compute({'input01': {data: input01Data}, 'input11': {data: input11Data}});
    utils.checkValue(outputs.output01.data, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test min converted from minimum_broadcast_int32 test', async function() {
    // Converted test case (from: V1_2/minimum.mod.py)
    const builder = new MLGraphBuilder(context);
    const input01 = builder.input('input01', {type: 'int32', dimensions: [3, 1, 2]});
    const input01Data = new Int32Array([1, 0, -1, -2, -1, 11]);
    const input11 = builder.input('input11', {type: 'int32', dimensions: [2]});
    const input11Data = new Int32Array([0, 2]);
    const expected = [0, 0, -1, -2, -1, 2];
    const output01 = builder.min(input01, input11);
    const graph = await builder.build({output01});
    const outputs = await graph.compute({'input01': {data: input01Data}, 'input11': {data: input11Data}});
    utils.checkValue(outputs.output01.data, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
