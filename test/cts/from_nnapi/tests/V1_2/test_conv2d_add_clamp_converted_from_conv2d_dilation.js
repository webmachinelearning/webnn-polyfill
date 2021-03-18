'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test conv2d + add + clamp converted from conv2d_dilation_nhwc test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
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
    const param7 = 1;
    const param8 = 1;
    const expected = [0.875, 0.875, 0.875, 0.875];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'dilations': [param8, param7], 'filterLayout': 'hwio'});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const model = builder.createModel({op4});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}});
    utils.checkValue(outputs.op4.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_dilation_nhwc_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
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
    const param7 = 1;
    const param8 = 1;
    const expected = [0.875, 0.875, 0.875, 0.875];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'dilations': [param8, param7], 'filterLayout': 'hwio'});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const model = builder.createModel({op4});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}});
    utils.checkValue(outputs.op4.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_dilation_nhwc_weight_as_input test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
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
    const param7 = 1;
    const param8 = 1;
    const expected = [0.875, 0.875, 0.875, 0.875];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'dilations': [param8, param7], 'filterLayout': 'hwio'});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const model = builder.createModel({op4});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}, 'op2': {buffer: op2Buffer}, 'op3': {buffer: op3Buffer}});
    utils.checkValue(outputs.op4.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_dilation_nhwc_weight_as_input_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
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
    const param7 = 1;
    const param8 = 1;
    const expected = [0.875, 0.875, 0.875, 0.875];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'dilations': [param8, param7], 'filterLayout': 'hwio'});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const model = builder.createModel({op4});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}, 'op2': {buffer: op2Buffer}, 'op3': {buffer: op3Buffer}});
    utils.checkValue(outputs.op4.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_dilation_nchw test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
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
    const param7 = 1;
    const param8 = 1;
    const expected = [0.875, 0.875, 0.875, 0.875];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'dilations': [param8, param7]});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const model = builder.createModel({op4});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}});
    utils.checkValue(outputs.op4.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_dilation_nchw_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
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
    const param7 = 1;
    const param8 = 1;
    const expected = [0.875, 0.875, 0.875, 0.875];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'dilations': [param8, param7]});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const model = builder.createModel({op4});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}});
    utils.checkValue(outputs.op4.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_dilation_nchw_weight_as_input test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
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
    const param7 = 1;
    const param8 = 1;
    const expected = [0.875, 0.875, 0.875, 0.875];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'dilations': [param8, param7]});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const model = builder.createModel({op4});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}, 'op2': {buffer: op2Buffer}, 'op3': {buffer: op3Buffer}});
    utils.checkValue(outputs.op4.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_dilation_nchw_weight_as_input_relaxed test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
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
    const param7 = 1;
    const param8 = 1;
    const expected = [0.875, 0.875, 0.875, 0.875];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'inputLayout': layout, 'dilations': [param8, param7]});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const model = builder.createModel({op4});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}, 'op2': {buffer: op2Buffer}, 'op3': {buffer: op3Buffer}});
    utils.checkValue(outputs.op4.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_dilation_nhwc_2 test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = nn.createModelBuilder();
    const op11 = builder.input('op11', {type: 'float32', dimensions: [1, 9, 9, 1]});
    const op11Buffer = new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const op21 = builder.constant({type: 'float32', dimensions: [3, 3, 1, 1]}, new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]));
    const op31 = builder.constant({type: 'float32', dimensions: [1]}, new Float32Array([0]));
    const param9 = 0;
    const param10 = 0;
    const param11 = 0;
    const param12 = 0;
    const param13 = 1;
    const param14 = 1;
    const layout = 'nhwc';
    const param16 = 3;
    const param17 = 3;
    const expected = [5, 5, 5, 5, 5, 5, 5, 5, 5];
    const interOut0 = builder.conv2d(op11, op21, {'padding': [param11, param12, param9, param10], 'strides': [param14, param13], 'inputLayout': layout, 'dilations': [param17, param16], 'filterLayout': 'hwio'});
    const interOut1 = builder.add(interOut0, op31);
    const op41 = builder.clamp(interOut1);
    const model = builder.createModel({op41});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op11': {buffer: op11Buffer}});
    utils.checkValue(outputs.op41.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_dilation_nhwc_relaxed_2 test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = nn.createModelBuilder();
    const op11 = builder.input('op11', {type: 'float32', dimensions: [1, 9, 9, 1]});
    const op11Buffer = new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const op21 = builder.constant({type: 'float32', dimensions: [3, 3, 1, 1]}, new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]));
    const op31 = builder.constant({type: 'float32', dimensions: [1]}, new Float32Array([0]));
    const param9 = 0;
    const param10 = 0;
    const param11 = 0;
    const param12 = 0;
    const param13 = 1;
    const param14 = 1;
    const layout = 'nhwc';
    const param16 = 3;
    const param17 = 3;
    const expected = [5, 5, 5, 5, 5, 5, 5, 5, 5];
    const interOut0 = builder.conv2d(op11, op21, {'padding': [param11, param12, param9, param10], 'strides': [param14, param13], 'inputLayout': layout, 'dilations': [param17, param16], 'filterLayout': 'hwio'});
    const interOut1 = builder.add(interOut0, op31);
    const op41 = builder.clamp(interOut1);
    const model = builder.createModel({op41});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op11': {buffer: op11Buffer}});
    utils.checkValue(outputs.op41.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_dilation_nhwc_weight_as_input_2 test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = nn.createModelBuilder();
    const op11 = builder.input('op11', {type: 'float32', dimensions: [1, 9, 9, 1]});
    const op11Buffer = new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const op21 = builder.input('op21', {type: 'float32', dimensions: [3, 3, 1, 1]});
    const op21Buffer = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const op31 = builder.input('op31', {type: 'float32', dimensions: [1]});
    const op31Buffer = new Float32Array([0]);
    const param9 = 0;
    const param10 = 0;
    const param11 = 0;
    const param12 = 0;
    const param13 = 1;
    const param14 = 1;
    const layout = 'nhwc';
    const param16 = 3;
    const param17 = 3;
    const expected = [5, 5, 5, 5, 5, 5, 5, 5, 5];
    const interOut0 = builder.conv2d(op11, op21, {'padding': [param11, param12, param9, param10], 'strides': [param14, param13], 'inputLayout': layout, 'dilations': [param17, param16], 'filterLayout': 'hwio'});
    const interOut1 = builder.add(interOut0, op31);
    const op41 = builder.clamp(interOut1);
    const model = builder.createModel({op41});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op11': {buffer: op11Buffer}, 'op21': {buffer: op21Buffer}, 'op31': {buffer: op31Buffer}});
    utils.checkValue(outputs.op41.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_dilation_nhwc_weight_as_input_relaxed_2 test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = nn.createModelBuilder();
    const op11 = builder.input('op11', {type: 'float32', dimensions: [1, 9, 9, 1]});
    const op11Buffer = new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const op21 = builder.input('op21', {type: 'float32', dimensions: [3, 3, 1, 1]});
    const op21Buffer = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const op31 = builder.input('op31', {type: 'float32', dimensions: [1]});
    const op31Buffer = new Float32Array([0]);
    const param9 = 0;
    const param10 = 0;
    const param11 = 0;
    const param12 = 0;
    const param13 = 1;
    const param14 = 1;
    const layout = 'nhwc';
    const param16 = 3;
    const param17 = 3;
    const expected = [5, 5, 5, 5, 5, 5, 5, 5, 5];
    const interOut0 = builder.conv2d(op11, op21, {'padding': [param11, param12, param9, param10], 'strides': [param14, param13], 'inputLayout': layout, 'dilations': [param17, param16], 'filterLayout': 'hwio'});
    const interOut1 = builder.add(interOut0, op31);
    const op41 = builder.clamp(interOut1);
    const model = builder.createModel({op41});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op11': {buffer: op11Buffer}, 'op21': {buffer: op21Buffer}, 'op31': {buffer: op31Buffer}});
    utils.checkValue(outputs.op41.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_dilation_nchw_2 test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = nn.createModelBuilder();
    const op11 = builder.input('op11', {type: 'float32', dimensions: [1, 1, 9, 9]});
    const op11Buffer = new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const op21 = builder.constant({type: 'float32', dimensions: [1, 1, 3, 3]}, new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]));
    const op31 = builder.constant({type: 'float32', dimensions: [1, 1, 1, 1]}, new Float32Array([0]));
    const param9 = 0;
    const param10 = 0;
    const param11 = 0;
    const param12 = 0;
    const param13 = 1;
    const param14 = 1;
    const layout = 'nchw';
    const param16 = 3;
    const param17 = 3;
    const expected = [5, 5, 5, 5, 5, 5, 5, 5, 5];
    const interOut0 = builder.conv2d(op11, op21, {'padding': [param11, param12, param9, param10], 'strides': [param14, param13], 'inputLayout': layout, 'dilations': [param17, param16]});
    const interOut1 = builder.add(interOut0, op31);
    const op41 = builder.clamp(interOut1);
    const model = builder.createModel({op41});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op11': {buffer: op11Buffer}});
    utils.checkValue(outputs.op41.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_dilation_nchw_relaxed_2 test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = nn.createModelBuilder();
    const op11 = builder.input('op11', {type: 'float32', dimensions: [1, 1, 9, 9]});
    const op11Buffer = new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const op21 = builder.constant({type: 'float32', dimensions: [1, 1, 3, 3]}, new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]));
    const op31 = builder.constant({type: 'float32', dimensions: [1, 1, 1, 1]}, new Float32Array([0]));
    const param9 = 0;
    const param10 = 0;
    const param11 = 0;
    const param12 = 0;
    const param13 = 1;
    const param14 = 1;
    const layout = 'nchw';
    const param16 = 3;
    const param17 = 3;
    const expected = [5, 5, 5, 5, 5, 5, 5, 5, 5];
    const interOut0 = builder.conv2d(op11, op21, {'padding': [param11, param12, param9, param10], 'strides': [param14, param13], 'inputLayout': layout, 'dilations': [param17, param16]});
    const interOut1 = builder.add(interOut0, op31);
    const op41 = builder.clamp(interOut1);
    const model = builder.createModel({op41});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op11': {buffer: op11Buffer}});
    utils.checkValue(outputs.op41.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_dilation_nchw_weight_as_input_2 test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = nn.createModelBuilder();
    const op11 = builder.input('op11', {type: 'float32', dimensions: [1, 1, 9, 9]});
    const op11Buffer = new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const op21 = builder.input('op21', {type: 'float32', dimensions: [1, 1, 3, 3]});
    const op21Buffer = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const op31 = builder.input('op31', {type: 'float32', dimensions: [1, 1, 1, 1]});
    const op31Buffer = new Float32Array([0]);
    const param9 = 0;
    const param10 = 0;
    const param11 = 0;
    const param12 = 0;
    const param13 = 1;
    const param14 = 1;
    const layout = 'nchw';
    const param16 = 3;
    const param17 = 3;
    const expected = [5, 5, 5, 5, 5, 5, 5, 5, 5];
    const interOut0 = builder.conv2d(op11, op21, {'padding': [param11, param12, param9, param10], 'strides': [param14, param13], 'inputLayout': layout, 'dilations': [param17, param16]});
    const interOut1 = builder.add(interOut0, op31);
    const op41 = builder.clamp(interOut1);
    const model = builder.createModel({op41});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op11': {buffer: op11Buffer}, 'op21': {buffer: op21Buffer}, 'op31': {buffer: op31Buffer}});
    utils.checkValue(outputs.op41.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test conv2d + add + clamp converted from conv2d_dilation_nchw_weight_as_input_relaxed_2 test', async function() {
    // Converted test case (from: V1_2/conv2d_dilation.mod.py)
    const builder = nn.createModelBuilder();
    const op11 = builder.input('op11', {type: 'float32', dimensions: [1, 1, 9, 9]});
    const op11Buffer = new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    const op21 = builder.input('op21', {type: 'float32', dimensions: [1, 1, 3, 3]});
    const op21Buffer = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const op31 = builder.input('op31', {type: 'float32', dimensions: [1, 1, 1, 1]});
    const op31Buffer = new Float32Array([0]);
    const param9 = 0;
    const param10 = 0;
    const param11 = 0;
    const param12 = 0;
    const param13 = 1;
    const param14 = 1;
    const layout = 'nchw';
    const param16 = 3;
    const param17 = 3;
    const expected = [5, 5, 5, 5, 5, 5, 5, 5, 5];
    const interOut0 = builder.conv2d(op11, op21, {'padding': [param11, param12, param9, param10], 'strides': [param14, param13], 'inputLayout': layout, 'dilations': [param17, param16]});
    const interOut1 = builder.add(interOut0, op31);
    const op41 = builder.clamp(interOut1);
    const model = builder.createModel({op41});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op11': {buffer: op11Buffer}, 'op21': {buffer: op21Buffer}, 'op31': {buffer: op31Buffer}});
    utils.checkValue(outputs.op41.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
