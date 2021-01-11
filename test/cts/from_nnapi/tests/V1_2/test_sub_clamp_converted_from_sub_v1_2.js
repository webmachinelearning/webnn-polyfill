'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test sub + clamp converted from sub_v1_2_none test', async function() {
    // Converted test case (from: V1_2/sub_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const input0 = builder.input('input0', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const input0Buffer = new Float32Array([2.0, -4.0, 8.0, -16.0]);
    const input1 = builder.input('input1', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const input1Buffer = new Float32Array([2.0, -2.0, -4.0, 4.0]);
    const expected = [0.0, -2.0, 12.0, -20.0];
    const interOut0 = builder.sub(input0, input1);
    const output0 = builder.clamp(interOut0);
    const model = builder.createModel({output0});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input0': {buffer: input0Buffer}, 'input1': {buffer: input1Buffer}});
    utils.checkValue(outputs.output0.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test sub + clamp converted from sub_v1_2_relu test', async function() {
    // Converted test case (from: V1_2/sub_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const input0 = builder.input('input0', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const input0Buffer = new Float32Array([2.0, -4.0, 8.0, -16.0]);
    const input1 = builder.input('input1', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const input1Buffer = new Float32Array([2.0, -2.0, -4.0, 4.0]);
    const expected = [0.0, 0.0, 12.0, 0.0];
    const interOut0 = builder.sub(input0, input1);
    const output0 = builder.relu(interOut0);
    const model = builder.createModel({output0});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input0': {buffer: input0Buffer}, 'input1': {buffer: input1Buffer}});
    utils.checkValue(outputs.output0.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test sub + clamp converted from sub_v1_2_relu1 test', async function() {
    // Converted test case (from: V1_2/sub_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const input0 = builder.input('input0', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const input0Buffer = new Float32Array([2.0, -4.0, 8.0, -16.0]);
    const input1 = builder.input('input1', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const input1Buffer = new Float32Array([2.0, -2.0, -4.0, 4.0]);
    const expected = [0.0, -1.0, 1.0, -1.0];
    const interOut0 = builder.sub(input0, input1);
    const output0 = builder.clamp(interOut0, {minValue: builder.constant(-1), maxValue: builder.constant(1)});
    const model = builder.createModel({output0});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input0': {buffer: input0Buffer}, 'input1': {buffer: input1Buffer}});
    utils.checkValue(outputs.output0.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test sub + clamp converted from sub_v1_2_relu6 test', async function() {
    // Converted test case (from: V1_2/sub_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const input0 = builder.input('input0', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const input0Buffer = new Float32Array([2.0, -4.0, 8.0, -16.0]);
    const input1 = builder.input('input1', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const input1Buffer = new Float32Array([2.0, -2.0, -4.0, 4.0]);
    const expected = [0.0, 0.0, 6.0, 0.0];
    const interOut0 = builder.sub(input0, input1);
    const output0 = builder.clamp(interOut0, {minValue: builder.constant(0), maxValue: builder.constant(6)});
    const model = builder.createModel({output0});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input0': {buffer: input0Buffer}, 'input1': {buffer: input1Buffer}});
    utils.checkValue(outputs.output0.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
