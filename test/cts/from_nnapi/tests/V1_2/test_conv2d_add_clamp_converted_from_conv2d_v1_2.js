'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test conv2d + add + clamp converted from conv2d_v1_2_nhwc test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 3, 3, 1]});
    const op1Buffer = new Float32Array([1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]);
    const op2 = builder.constant({type: 'float32', dimensions: [2, 2, 1, 1]}, new Float32Array([0.25, 0.25, 0.25, 0.25]));
    const op3 = builder.constant({type: 'float32', dimensions: [1]}, new Float32Array([0]));
    const param = 0;
    const param1 = 0;
    const param2 = 0;
    const param3 = 0;
    const param4 = 1;
    const param5 = 1;
    const layout = 'nhwc';
    const expected = [0.875, 0.875, 0.875, 0.875];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'layout': layout});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const model = builder.createModel({op4});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}});
    utils.checkValue(outputs.op4.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_nhwc_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 3, 3, 1]});
    const op1Buffer = new Float32Array([1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]);
    const op2 = builder.constant({type: 'float32', dimensions: [2, 2, 1, 1]}, new Float32Array([0.25, 0.25, 0.25, 0.25]));
    const op3 = builder.constant({type: 'float32', dimensions: [1]}, new Float32Array([0]));
    const param = 0;
    const param1 = 0;
    const param2 = 0;
    const param3 = 0;
    const param4 = 1;
    const param5 = 1;
    const layout = 'nhwc';
    const expected = [0.875, 0.875, 0.875, 0.875];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'layout': layout});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const model = builder.createModel({op4});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}});
    utils.checkValue(outputs.op4.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_nhwc_weight_as_input test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 3, 3, 1]});
    const op1Buffer = new Float32Array([1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]);
    const op2 = builder.input('op2', {type: 'float32', dimensions: [2, 2, 1, 1]});
    const op2Buffer = new Float32Array([0.25, 0.25, 0.25, 0.25]);
    const op3 = builder.input('op3', {type: 'float32', dimensions: [1]});
    const op3Buffer = new Float32Array([0]);
    const param = 0;
    const param1 = 0;
    const param2 = 0;
    const param3 = 0;
    const param4 = 1;
    const param5 = 1;
    const layout = 'nhwc';
    const expected = [0.875, 0.875, 0.875, 0.875];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'layout': layout});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const model = builder.createModel({op4});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}, 'op2': {buffer: op2Buffer}, 'op3': {buffer: op3Buffer}});
    utils.checkValue(outputs.op4.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_nhwc_weight_as_input_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 3, 3, 1]});
    const op1Buffer = new Float32Array([1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]);
    const op2 = builder.input('op2', {type: 'float32', dimensions: [2, 2, 1, 1]});
    const op2Buffer = new Float32Array([0.25, 0.25, 0.25, 0.25]);
    const op3 = builder.input('op3', {type: 'float32', dimensions: [1]});
    const op3Buffer = new Float32Array([0]);
    const param = 0;
    const param1 = 0;
    const param2 = 0;
    const param3 = 0;
    const param4 = 1;
    const param5 = 1;
    const layout = 'nhwc';
    const expected = [0.875, 0.875, 0.875, 0.875];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'layout': layout});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const model = builder.createModel({op4});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}, 'op2': {buffer: op2Buffer}, 'op3': {buffer: op3Buffer}});
    utils.checkValue(outputs.op4.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_nchw test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 1, 3, 3]});
    const op1Buffer = new Float32Array([1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]);
    const op2 = builder.constant({type: 'float32', dimensions: [1, 1, 2, 2]}, new Float32Array([0.25, 0.25, 0.25, 0.25]));
    const op3 = builder.constant({type: 'float32', dimensions: [1, 1, 1, 1]}, new Float32Array([0]));
    const param = 0;
    const param1 = 0;
    const param2 = 0;
    const param3 = 0;
    const param4 = 1;
    const param5 = 1;
    const layout = 'nchw';
    const expected = [0.875, 0.875, 0.875, 0.875];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'layout': layout});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const model = builder.createModel({op4});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}});
    utils.checkValue(outputs.op4.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_nchw_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 1, 3, 3]});
    const op1Buffer = new Float32Array([1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]);
    const op2 = builder.constant({type: 'float32', dimensions: [1, 1, 2, 2]}, new Float32Array([0.25, 0.25, 0.25, 0.25]));
    const op3 = builder.constant({type: 'float32', dimensions: [1, 1, 1, 1]}, new Float32Array([0]));
    const param = 0;
    const param1 = 0;
    const param2 = 0;
    const param3 = 0;
    const param4 = 1;
    const param5 = 1;
    const layout = 'nchw';
    const expected = [0.875, 0.875, 0.875, 0.875];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'layout': layout});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const model = builder.createModel({op4});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}});
    utils.checkValue(outputs.op4.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_nchw_weight_as_input test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 1, 3, 3]});
    const op1Buffer = new Float32Array([1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]);
    const op2 = builder.input('op2', {type: 'float32', dimensions: [1, 1, 2, 2]});
    const op2Buffer = new Float32Array([0.25, 0.25, 0.25, 0.25]);
    const op3 = builder.input('op3', {type: 'float32', dimensions: [1, 1, 1, 1]});
    const op3Buffer = new Float32Array([0]);
    const param = 0;
    const param1 = 0;
    const param2 = 0;
    const param3 = 0;
    const param4 = 1;
    const param5 = 1;
    const layout = 'nchw';
    const expected = [0.875, 0.875, 0.875, 0.875];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'layout': layout});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const model = builder.createModel({op4});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}, 'op2': {buffer: op2Buffer}, 'op3': {buffer: op3Buffer}});
    utils.checkValue(outputs.op4.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_nchw_weight_as_input_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 1, 3, 3]});
    const op1Buffer = new Float32Array([1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]);
    const op2 = builder.input('op2', {type: 'float32', dimensions: [1, 1, 2, 2]});
    const op2Buffer = new Float32Array([0.25, 0.25, 0.25, 0.25]);
    const op3 = builder.input('op3', {type: 'float32', dimensions: [1, 1, 1, 1]});
    const op3Buffer = new Float32Array([0]);
    const param = 0;
    const param1 = 0;
    const param2 = 0;
    const param3 = 0;
    const param4 = 1;
    const param5 = 1;
    const layout = 'nchw';
    const expected = [0.875, 0.875, 0.875, 0.875];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'layout': layout});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const model = builder.createModel({op4});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}, 'op2': {buffer: op2Buffer}, 'op3': {buffer: op3Buffer}});
    utils.checkValue(outputs.op4.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_channel_nhwc test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 1, 1, 3]});
    const op12Buffer = new Float32Array([5.0, 5.0, 5.0]);
    const op22 = builder.constant({type: 'float32', dimensions: [1, 1, 3, 3]}, new Float32Array([0.5, 2.0, 3.5, 1.0, 2.5, 4.0, 1.5, 3.0, 4.5]));
    const op32 = builder.constant({type: 'float32', dimensions: [3]}, new Float32Array([0.0, 0.0, 0.0]));
    const param11 = 0;
    const param12 = 0;
    const param13 = 0;
    const param14 = 0;
    const param15 = 1;
    const param16 = 1;
    const layout = 'nhwc';
    const expected = [15.0, 37.5, 60.0];
    const interOut0 = builder.conv2d(op12, op22, {'padding': [param13, param14, param11, param12], 'strides': [param16, param15], 'layout': layout});
    const interOut1 = builder.add(interOut0, op32);
    const op42 = builder.clamp(interOut1);
    const model = builder.createModel({op42});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op12': {buffer: op12Buffer}});
    utils.checkValue(outputs.op42.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_channel_nhwc_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 1, 1, 3]});
    const op12Buffer = new Float32Array([5.0, 5.0, 5.0]);
    const op22 = builder.constant({type: 'float32', dimensions: [1, 1, 3, 3]}, new Float32Array([0.5, 2.0, 3.5, 1.0, 2.5, 4.0, 1.5, 3.0, 4.5]));
    const op32 = builder.constant({type: 'float32', dimensions: [3]}, new Float32Array([0.0, 0.0, 0.0]));
    const param11 = 0;
    const param12 = 0;
    const param13 = 0;
    const param14 = 0;
    const param15 = 1;
    const param16 = 1;
    const layout = 'nhwc';
    const expected = [15.0, 37.5, 60.0];
    const interOut0 = builder.conv2d(op12, op22, {'padding': [param13, param14, param11, param12], 'strides': [param16, param15], 'layout': layout});
    const interOut1 = builder.add(interOut0, op32);
    const op42 = builder.clamp(interOut1);
    const model = builder.createModel({op42});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op12': {buffer: op12Buffer}});
    utils.checkValue(outputs.op42.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_channel_nhwc_weight_as_input test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 1, 1, 3]});
    const op12Buffer = new Float32Array([5.0, 5.0, 5.0]);
    const op22 = builder.input('op22', {type: 'float32', dimensions: [1, 1, 3, 3]});
    const op22Buffer = new Float32Array([0.5, 2.0, 3.5, 1.0, 2.5, 4.0, 1.5, 3.0, 4.5]);
    const op32 = builder.input('op32', {type: 'float32', dimensions: [3]});
    const op32Buffer = new Float32Array([0.0, 0.0, 0.0]);
    const param11 = 0;
    const param12 = 0;
    const param13 = 0;
    const param14 = 0;
    const param15 = 1;
    const param16 = 1;
    const layout = 'nhwc';
    const expected = [15.0, 37.5, 60.0];
    const interOut0 = builder.conv2d(op12, op22, {'padding': [param13, param14, param11, param12], 'strides': [param16, param15], 'layout': layout});
    const interOut1 = builder.add(interOut0, op32);
    const op42 = builder.clamp(interOut1);
    const model = builder.createModel({op42});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op12': {buffer: op12Buffer}, 'op22': {buffer: op22Buffer}, 'op32': {buffer: op32Buffer}});
    utils.checkValue(outputs.op42.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_channel_nhwc_weight_as_input_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 1, 1, 3]});
    const op12Buffer = new Float32Array([5.0, 5.0, 5.0]);
    const op22 = builder.input('op22', {type: 'float32', dimensions: [1, 1, 3, 3]});
    const op22Buffer = new Float32Array([0.5, 2.0, 3.5, 1.0, 2.5, 4.0, 1.5, 3.0, 4.5]);
    const op32 = builder.input('op32', {type: 'float32', dimensions: [3]});
    const op32Buffer = new Float32Array([0.0, 0.0, 0.0]);
    const param11 = 0;
    const param12 = 0;
    const param13 = 0;
    const param14 = 0;
    const param15 = 1;
    const param16 = 1;
    const layout = 'nhwc';
    const expected = [15.0, 37.5, 60.0];
    const interOut0 = builder.conv2d(op12, op22, {'padding': [param13, param14, param11, param12], 'strides': [param16, param15], 'layout': layout});
    const interOut1 = builder.add(interOut0, op32);
    const op42 = builder.clamp(interOut1);
    const model = builder.createModel({op42});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op12': {buffer: op12Buffer}, 'op22': {buffer: op22Buffer}, 'op32': {buffer: op32Buffer}});
    utils.checkValue(outputs.op42.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_channel_nchw test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 3, 1, 1]});
    const op12Buffer = new Float32Array([5.0, 5.0, 5.0]);
    const op22 = builder.constant({type: 'float32', dimensions: [3, 3, 1, 1]}, new Float32Array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]));
    const op32 = builder.constant({type: 'float32', dimensions: [1, 3, 1, 1]}, new Float32Array([0.0, 0.0, 0.0]));
    const param11 = 0;
    const param12 = 0;
    const param13 = 0;
    const param14 = 0;
    const param15 = 1;
    const param16 = 1;
    const layout = 'nchw';
    const expected = [15.0, 37.5, 60.0];
    const interOut0 = builder.conv2d(op12, op22, {'padding': [param13, param14, param11, param12], 'strides': [param16, param15], 'layout': layout});
    const interOut1 = builder.add(interOut0, op32);
    const op42 = builder.clamp(interOut1);
    const model = builder.createModel({op42});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op12': {buffer: op12Buffer}});
    utils.checkValue(outputs.op42.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_channel_nchw_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 3, 1, 1]});
    const op12Buffer = new Float32Array([5.0, 5.0, 5.0]);
    const op22 = builder.constant({type: 'float32', dimensions: [3, 3, 1, 1]}, new Float32Array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]));
    const op32 = builder.constant({type: 'float32', dimensions: [1, 3, 1, 1]}, new Float32Array([0.0, 0.0, 0.0]));
    const param11 = 0;
    const param12 = 0;
    const param13 = 0;
    const param14 = 0;
    const param15 = 1;
    const param16 = 1;
    const layout = 'nchw';
    const expected = [15.0, 37.5, 60.0];
    const interOut0 = builder.conv2d(op12, op22, {'padding': [param13, param14, param11, param12], 'strides': [param16, param15], 'layout': layout});
    const interOut1 = builder.add(interOut0, op32);
    const op42 = builder.clamp(interOut1);
    const model = builder.createModel({op42});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op12': {buffer: op12Buffer}});
    utils.checkValue(outputs.op42.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_channel_nchw_weight_as_input test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 3, 1, 1]});
    const op12Buffer = new Float32Array([5.0, 5.0, 5.0]);
    const op22 = builder.input('op22', {type: 'float32', dimensions: [3, 3, 1, 1]});
    const op22Buffer = new Float32Array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]);
    const op32 = builder.input('op32', {type: 'float32', dimensions: [1, 3, 1, 1]});
    const op32Buffer = new Float32Array([0.0, 0.0, 0.0]);
    const param11 = 0;
    const param12 = 0;
    const param13 = 0;
    const param14 = 0;
    const param15 = 1;
    const param16 = 1;
    const layout = 'nchw';
    const expected = [15.0, 37.5, 60.0];
    const interOut0 = builder.conv2d(op12, op22, {'padding': [param13, param14, param11, param12], 'strides': [param16, param15], 'layout': layout});
    const interOut1 = builder.add(interOut0, op32);
    const op42 = builder.clamp(interOut1);
    const model = builder.createModel({op42});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op12': {buffer: op12Buffer}, 'op22': {buffer: op22Buffer}, 'op32': {buffer: op32Buffer}});
    utils.checkValue(outputs.op42.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_channel_nchw_weight_as_input_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 3, 1, 1]});
    const op12Buffer = new Float32Array([5.0, 5.0, 5.0]);
    const op22 = builder.input('op22', {type: 'float32', dimensions: [3, 3, 1, 1]});
    const op22Buffer = new Float32Array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]);
    const op32 = builder.input('op32', {type: 'float32', dimensions: [1, 3, 1, 1]});
    const op32Buffer = new Float32Array([0.0, 0.0, 0.0]);
    const param11 = 0;
    const param12 = 0;
    const param13 = 0;
    const param14 = 0;
    const param15 = 1;
    const param16 = 1;
    const layout = 'nchw';
    const expected = [15.0, 37.5, 60.0];
    const interOut0 = builder.conv2d(op12, op22, {'padding': [param13, param14, param11, param12], 'strides': [param16, param15], 'layout': layout});
    const interOut1 = builder.add(interOut0, op32);
    const op42 = builder.clamp(interOut1);
    const model = builder.createModel({op42});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op12': {buffer: op12Buffer}, 'op22': {buffer: op22Buffer}, 'op32': {buffer: op32Buffer}});
    utils.checkValue(outputs.op42.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_large_nhwc test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 2, 3, 3]});
    const op13Buffer = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]);
    const op23 = builder.constant({type: 'float32', dimensions: [1, 1, 3, 3]}, new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]));
    const op33 = builder.constant({type: 'float32', dimensions: [3]}, new Float32Array([0.0, 0.0, 0.0]));
    const param18 = 0;
    const param19 = 0;
    const param20 = 0;
    const param21 = 0;
    const param22 = 1;
    const param23 = 1;
    const layout = 'nhwc';
    const expected = [30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0, 204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0];
    const interOut0 = builder.conv2d(op13, op23, {'padding': [param20, param21, param18, param19], 'strides': [param23, param22], 'layout': layout});
    const interOut1 = builder.add(interOut0, op33);
    const op43 = builder.clamp(interOut1);
    const model = builder.createModel({op43});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op13': {buffer: op13Buffer}});
    utils.checkValue(outputs.op43.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_large_nhwc_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 2, 3, 3]});
    const op13Buffer = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]);
    const op23 = builder.constant({type: 'float32', dimensions: [1, 1, 3, 3]}, new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]));
    const op33 = builder.constant({type: 'float32', dimensions: [3]}, new Float32Array([0.0, 0.0, 0.0]));
    const param18 = 0;
    const param19 = 0;
    const param20 = 0;
    const param21 = 0;
    const param22 = 1;
    const param23 = 1;
    const layout = 'nhwc';
    const expected = [30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0, 204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0];
    const interOut0 = builder.conv2d(op13, op23, {'padding': [param20, param21, param18, param19], 'strides': [param23, param22], 'layout': layout});
    const interOut1 = builder.add(interOut0, op33);
    const op43 = builder.clamp(interOut1);
    const model = builder.createModel({op43});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op13': {buffer: op13Buffer}});
    utils.checkValue(outputs.op43.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_large_nhwc_weight_as_input test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 2, 3, 3]});
    const op13Buffer = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]);
    const op23 = builder.input('op23', {type: 'float32', dimensions: [1, 1, 3, 3]});
    const op23Buffer = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    const op33 = builder.input('op33', {type: 'float32', dimensions: [3]});
    const op33Buffer = new Float32Array([0.0, 0.0, 0.0]);
    const param18 = 0;
    const param19 = 0;
    const param20 = 0;
    const param21 = 0;
    const param22 = 1;
    const param23 = 1;
    const layout = 'nhwc';
    const expected = [30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0, 204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0];
    const interOut0 = builder.conv2d(op13, op23, {'padding': [param20, param21, param18, param19], 'strides': [param23, param22], 'layout': layout});
    const interOut1 = builder.add(interOut0, op33);
    const op43 = builder.clamp(interOut1);
    const model = builder.createModel({op43});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op13': {buffer: op13Buffer}, 'op23': {buffer: op23Buffer}, 'op33': {buffer: op33Buffer}});
    utils.checkValue(outputs.op43.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_large_nhwc_weight_as_input_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 2, 3, 3]});
    const op13Buffer = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]);
    const op23 = builder.input('op23', {type: 'float32', dimensions: [1, 1, 3, 3]});
    const op23Buffer = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    const op33 = builder.input('op33', {type: 'float32', dimensions: [3]});
    const op33Buffer = new Float32Array([0.0, 0.0, 0.0]);
    const param18 = 0;
    const param19 = 0;
    const param20 = 0;
    const param21 = 0;
    const param22 = 1;
    const param23 = 1;
    const layout = 'nhwc';
    const expected = [30.0, 36.0, 42.0, 66.0, 81.0, 96.0, 102.0, 126.0, 150.0, 138.0, 171.0, 204.0, 174.0, 216.0, 258.0, 210.0, 261.0, 312.0];
    const interOut0 = builder.conv2d(op13, op23, {'padding': [param20, param21, param18, param19], 'strides': [param23, param22], 'layout': layout});
    const interOut1 = builder.add(interOut0, op33);
    const op43 = builder.clamp(interOut1);
    const model = builder.createModel({op43});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op13': {buffer: op13Buffer}, 'op23': {buffer: op23Buffer}, 'op33': {buffer: op33Buffer}});
    utils.checkValue(outputs.op43.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_large_nchw test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 3, 2, 3]});
    const op13Buffer = new Float32Array([1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0]);
    const op23 = builder.constant({type: 'float32', dimensions: [3, 3, 1, 1]}, new Float32Array([1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]));
    const op33 = builder.constant({type: 'float32', dimensions: [1, 3, 1, 1]}, new Float32Array([0.0, 0.0, 0.0]));
    const param18 = 0;
    const param19 = 0;
    const param20 = 0;
    const param21 = 0;
    const param22 = 1;
    const param23 = 1;
    const layout = 'nchw';
    const expected = [30.0, 66.0, 102.0, 138.0, 174.0, 210.0, 36.0, 81.0, 126.0, 171.0, 216.0, 261.0, 42.0, 96.0, 150.0, 204.0, 258.0, 312.0];
    const interOut0 = builder.conv2d(op13, op23, {'padding': [param20, param21, param18, param19], 'strides': [param23, param22], 'layout': layout});
    const interOut1 = builder.add(interOut0, op33);
    const op43 = builder.clamp(interOut1);
    const model = builder.createModel({op43});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op13': {buffer: op13Buffer}});
    utils.checkValue(outputs.op43.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_large_nchw_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 3, 2, 3]});
    const op13Buffer = new Float32Array([1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0]);
    const op23 = builder.constant({type: 'float32', dimensions: [3, 3, 1, 1]}, new Float32Array([1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]));
    const op33 = builder.constant({type: 'float32', dimensions: [1, 3, 1, 1]}, new Float32Array([0.0, 0.0, 0.0]));
    const param18 = 0;
    const param19 = 0;
    const param20 = 0;
    const param21 = 0;
    const param22 = 1;
    const param23 = 1;
    const layout = 'nchw';
    const expected = [30.0, 66.0, 102.0, 138.0, 174.0, 210.0, 36.0, 81.0, 126.0, 171.0, 216.0, 261.0, 42.0, 96.0, 150.0, 204.0, 258.0, 312.0];
    const interOut0 = builder.conv2d(op13, op23, {'padding': [param20, param21, param18, param19], 'strides': [param23, param22], 'layout': layout});
    const interOut1 = builder.add(interOut0, op33);
    const op43 = builder.clamp(interOut1);
    const model = builder.createModel({op43});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op13': {buffer: op13Buffer}});
    utils.checkValue(outputs.op43.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_large_nchw_weight_as_input test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 3, 2, 3]});
    const op13Buffer = new Float32Array([1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0]);
    const op23 = builder.input('op23', {type: 'float32', dimensions: [3, 3, 1, 1]});
    const op23Buffer = new Float32Array([1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]);
    const op33 = builder.input('op33', {type: 'float32', dimensions: [1, 3, 1, 1]});
    const op33Buffer = new Float32Array([0.0, 0.0, 0.0]);
    const param18 = 0;
    const param19 = 0;
    const param20 = 0;
    const param21 = 0;
    const param22 = 1;
    const param23 = 1;
    const layout = 'nchw';
    const expected = [30.0, 66.0, 102.0, 138.0, 174.0, 210.0, 36.0, 81.0, 126.0, 171.0, 216.0, 261.0, 42.0, 96.0, 150.0, 204.0, 258.0, 312.0];
    const interOut0 = builder.conv2d(op13, op23, {'padding': [param20, param21, param18, param19], 'strides': [param23, param22], 'layout': layout});
    const interOut1 = builder.add(interOut0, op33);
    const op43 = builder.clamp(interOut1);
    const model = builder.createModel({op43});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op13': {buffer: op13Buffer}, 'op23': {buffer: op23Buffer}, 'op33': {buffer: op33Buffer}});
    utils.checkValue(outputs.op43.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_v1_2_large_nchw_weight_as_input_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const op13 = builder.input('op13', {type: 'float32', dimensions: [1, 3, 2, 3]});
    const op13Buffer = new Float32Array([1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0]);
    const op23 = builder.input('op23', {type: 'float32', dimensions: [3, 3, 1, 1]});
    const op23Buffer = new Float32Array([1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]);
    const op33 = builder.input('op33', {type: 'float32', dimensions: [1, 3, 1, 1]});
    const op33Buffer = new Float32Array([0.0, 0.0, 0.0]);
    const param18 = 0;
    const param19 = 0;
    const param20 = 0;
    const param21 = 0;
    const param22 = 1;
    const param23 = 1;
    const layout = 'nchw';
    const expected = [30.0, 66.0, 102.0, 138.0, 174.0, 210.0, 36.0, 81.0, 126.0, 171.0, 216.0, 261.0, 42.0, 96.0, 150.0, 204.0, 258.0, 312.0];
    const interOut0 = builder.conv2d(op13, op23, {'padding': [param20, param21, param18, param19], 'strides': [param23, param22], 'layout': layout});
    const interOut1 = builder.add(interOut0, op33);
    const op43 = builder.clamp(interOut1);
    const model = builder.createModel({op43});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op13': {buffer: op13Buffer}, 'op23': {buffer: op23Buffer}, 'op33': {buffer: op33Buffer}});
    utils.checkValue(outputs.op43.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
