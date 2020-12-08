'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test min converted from minimum_simple test', async function() {
    // Converted test case (from: V1_2/minimum.mod.py)
    const builder = nn.createModelBuilder();
    const input0 = builder.input('input0', {type: 'float32', dimensions: [3, 1, 2]});
    const input0Buffer = new Float32Array([1.0, 0.0, -1.0, 11.0, -2.0, -1.44]);
    const input1 = builder.input('input1', {type: 'float32', dimensions: [3, 1, 2]});
    const input1Buffer = new Float32Array([-1.0, 0.0, 1.0, 12.0, -3.0, -1.43]);
    const expected = [-1.0, 0.0, -1.0, 11.0, -3.0, -1.44];
    const output0 = builder.min(input0, input1);
    const model = builder.createModel({output0});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input0': {buffer: input0Buffer}, 'input1': {buffer: input1Buffer}});
    utils.checkValue(outputs.output0.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test min converted from minimum_simple_relaxed test', async function() {
    // Converted test case (from: V1_2/minimum.mod.py)
    const builder = nn.createModelBuilder();
    const input0 = builder.input('input0', {type: 'float32', dimensions: [3, 1, 2]});
    const input0Buffer = new Float32Array([1.0, 0.0, -1.0, 11.0, -2.0, -1.44]);
    const input1 = builder.input('input1', {type: 'float32', dimensions: [3, 1, 2]});
    const input1Buffer = new Float32Array([-1.0, 0.0, 1.0, 12.0, -3.0, -1.43]);
    const expected = [-1.0, 0.0, -1.0, 11.0, -3.0, -1.44];
    const output0 = builder.min(input0, input1);
    const model = builder.createModel({output0});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input0': {buffer: input0Buffer}, 'input1': {buffer: input1Buffer}});
    utils.checkValue(outputs.output0.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test min converted from minimum_simple_int32 test', async function() {
    // Converted test case (from: V1_2/minimum.mod.py)
    const builder = nn.createModelBuilder();
    const input0 = builder.input('input0', {type: 'int32', dimensions: [3, 1, 2]});
    const input0Buffer = new Int32Array([1, 0, -1, 11, -2, -1]);
    const input1 = builder.input('input1', {type: 'int32', dimensions: [3, 1, 2]});
    const input1Buffer = new Int32Array([-1, 0, 1, 12, -3, -1]);
    const expected = [-1, 0, -1, 11, -3, -1];
    const output0 = builder.min(input0, input1);
    const model = builder.createModel({output0});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input0': {buffer: input0Buffer}, 'input1': {buffer: input1Buffer}});
    utils.checkValue(outputs.output0.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test min converted from minimum_broadcast test', async function() {
    // Converted test case (from: V1_2/minimum.mod.py)
    const builder = nn.createModelBuilder();
    const input01 = builder.input('input01', {type: 'float32', dimensions: [3, 1, 2]});
    const input01Buffer = new Float32Array([1.0, 0.0, -1.0, -2.0, -1.44, 11.0]);
    const input11 = builder.input('input11', {type: 'float32', dimensions: [2]});
    const input11Buffer = new Float32Array([0.5, 2.0]);
    const expected = [0.5, 0.0, -1.0, -2.0, -1.44, 2.0];
    const output01 = builder.min(input01, input11);
    const model = builder.createModel({output01});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input01': {buffer: input01Buffer}, 'input11': {buffer: input11Buffer}});
    utils.checkValue(outputs.output01.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test min converted from minimum_broadcast_relaxed test', async function() {
    // Converted test case (from: V1_2/minimum.mod.py)
    const builder = nn.createModelBuilder();
    const input01 = builder.input('input01', {type: 'float32', dimensions: [3, 1, 2]});
    const input01Buffer = new Float32Array([1.0, 0.0, -1.0, -2.0, -1.44, 11.0]);
    const input11 = builder.input('input11', {type: 'float32', dimensions: [2]});
    const input11Buffer = new Float32Array([0.5, 2.0]);
    const expected = [0.5, 0.0, -1.0, -2.0, -1.44, 2.0];
    const output01 = builder.min(input01, input11);
    const model = builder.createModel({output01});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input01': {buffer: input01Buffer}, 'input11': {buffer: input11Buffer}});
    utils.checkValue(outputs.output01.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test min converted from minimum_broadcast_int32 test', async function() {
    // Converted test case (from: V1_2/minimum.mod.py)
    const builder = nn.createModelBuilder();
    const input01 = builder.input('input01', {type: 'int32', dimensions: [3, 1, 2]});
    const input01Buffer = new Int32Array([1, 0, -1, -2, -1, 11]);
    const input11 = builder.input('input11', {type: 'int32', dimensions: [2]});
    const input11Buffer = new Int32Array([0, 2]);
    const expected = [0, 0, -1, -2, -1, 2];
    const output01 = builder.min(input01, input11);
    const model = builder.createModel({output01});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input01': {buffer: input01Buffer}, 'input11': {buffer: input11Buffer}});
    utils.checkValue(outputs.output01.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
