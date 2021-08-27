'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test l2Pool2d + clamp converted from l2_pool_v1_2_nhwc test', function() {
    // Converted test case (from: V1_2/l2_pool_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const op1Data = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const param = 0;
    const param1 = 0;
    const param2 = 0;
    const param3 = 0;
    const param4 = 1;
    const param5 = 1;
    const param6 = 1;
    const param7 = 1;
    const layout = 'nhwc';
    const expected = [1.0, 2.0, 3.0, 4.0];
    const interOut0 = builder.l2Pool2d(op1, {'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'windowDimensions': [param7, param6], 'layout': layout});
    const op4 = builder.clamp(interOut0);
    const graph = builder.build({op4});
    const outputs = {op4: new Float32Array(utils.sizeOfShape([1, 2, 2, 1]))};
    graph.compute({'op1': op1Data}, outputs);
    utils.checkValue(outputs.op4, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test l2Pool2d + clamp converted from l2_pool_v1_2_nhwc_relaxed test', function() {
    // Converted test case (from: V1_2/l2_pool_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const op1Data = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const param = 0;
    const param1 = 0;
    const param2 = 0;
    const param3 = 0;
    const param4 = 1;
    const param5 = 1;
    const param6 = 1;
    const param7 = 1;
    const layout = 'nhwc';
    const expected = [1.0, 2.0, 3.0, 4.0];
    const interOut0 = builder.l2Pool2d(op1, {'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'windowDimensions': [param7, param6], 'layout': layout});
    const op4 = builder.clamp(interOut0);
    const graph = builder.build({op4});
    const outputs = {op4: new Float32Array(utils.sizeOfShape([1, 2, 2, 1]))};
    graph.compute({'op1': op1Data}, outputs);
    utils.checkValue(outputs.op4, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test l2Pool2d + clamp converted from l2_pool_v1_2_nchw test', function() {
    // Converted test case (from: V1_2/l2_pool_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 1, 2, 2]});
    const op1Data = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const param = 0;
    const param1 = 0;
    const param2 = 0;
    const param3 = 0;
    const param4 = 1;
    const param5 = 1;
    const param6 = 1;
    const param7 = 1;
    const layout = 'nchw';
    const expected = [1.0, 2.0, 3.0, 4.0];
    const interOut0 = builder.l2Pool2d(op1, {'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'windowDimensions': [param7, param6], 'layout': layout});
    const op4 = builder.clamp(interOut0);
    const graph = builder.build({op4});
    const outputs = {op4: new Float32Array(utils.sizeOfShape([1, 1, 2, 2]))};
    graph.compute({'op1': op1Data}, outputs);
    utils.checkValue(outputs.op4, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test l2Pool2d + clamp converted from l2_pool_v1_2_nchw_relaxed test', function() {
    // Converted test case (from: V1_2/l2_pool_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 1, 2, 2]});
    const op1Data = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const param = 0;
    const param1 = 0;
    const param2 = 0;
    const param3 = 0;
    const param4 = 1;
    const param5 = 1;
    const param6 = 1;
    const param7 = 1;
    const layout = 'nchw';
    const expected = [1.0, 2.0, 3.0, 4.0];
    const interOut0 = builder.l2Pool2d(op1, {'padding': [param2, param3, param, param1], 'strides': [param5, param4], 'windowDimensions': [param7, param6], 'layout': layout});
    const op4 = builder.clamp(interOut0);
    const graph = builder.build({op4});
    const outputs = {op4: new Float32Array(utils.sizeOfShape([1, 1, 2, 2]))};
    graph.compute({'op1': op1Data}, outputs);
    utils.checkValue(outputs.op4, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test l2Pool2d + clamp converted from l2_pool_v1_2_large_nhwc test', function() {
    // Converted test case (from: V1_2/l2_pool_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 2, 2, 3]});
    const op12Data = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    const param15 = 0;
    const param16 = 0;
    const param17 = 0;
    const param18 = 0;
    const param19 = 1;
    const param20 = 1;
    const param21 = 2;
    const param22 = 2;
    const layout = 'nhwc';
    const expected = [6.442049503326416, 7.314369201660156, 8.215838432312012];
    const interOut0 = builder.l2Pool2d(op12, {'padding': [param17, param18, param15, param16], 'strides': [param20, param19], 'windowDimensions': [param22, param21], 'layout': layout});
    const op42 = builder.clamp(interOut0);
    const graph = builder.build({op42});
    const outputs = {op42: new Float32Array(utils.sizeOfShape([1, 1, 1, 3]))};
    graph.compute({'op12': op12Data}, outputs);
    utils.checkValue(outputs.op42, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test l2Pool2d + clamp converted from l2_pool_v1_2_large_nhwc_relaxed test', function() {
    // Converted test case (from: V1_2/l2_pool_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 2, 2, 3]});
    const op12Data = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    const param15 = 0;
    const param16 = 0;
    const param17 = 0;
    const param18 = 0;
    const param19 = 1;
    const param20 = 1;
    const param21 = 2;
    const param22 = 2;
    const layout = 'nhwc';
    const expected = [6.442049503326416, 7.314369201660156, 8.215838432312012];
    const interOut0 = builder.l2Pool2d(op12, {'padding': [param17, param18, param15, param16], 'strides': [param20, param19], 'windowDimensions': [param22, param21], 'layout': layout});
    const op42 = builder.clamp(interOut0);
    const graph = builder.build({op42});
    const outputs = {op42: new Float32Array(utils.sizeOfShape([1, 1, 1, 3]))};
    graph.compute({'op12': op12Data}, outputs);
    utils.checkValue(outputs.op42, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test l2Pool2d + clamp converted from l2_pool_v1_2_large_nchw test', function() {
    // Converted test case (from: V1_2/l2_pool_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 3, 2, 2]});
    const op12Data = new Float32Array([1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0]);
    const param15 = 0;
    const param16 = 0;
    const param17 = 0;
    const param18 = 0;
    const param19 = 1;
    const param20 = 1;
    const param21 = 2;
    const param22 = 2;
    const layout = 'nchw';
    const expected = [6.442049503326416, 7.314369201660156, 8.215838432312012];
    const interOut0 = builder.l2Pool2d(op12, {'padding': [param17, param18, param15, param16], 'strides': [param20, param19], 'windowDimensions': [param22, param21], 'layout': layout});
    const op42 = builder.clamp(interOut0);
    const graph = builder.build({op42});
    const outputs = {op42: new Float32Array(utils.sizeOfShape([1, 3, 1, 1]))};
    graph.compute({'op12': op12Data}, outputs);
    utils.checkValue(outputs.op42, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test l2Pool2d + clamp converted from l2_pool_v1_2_large_nchw_relaxed test', function() {
    // Converted test case (from: V1_2/l2_pool_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op12 = builder.input('op12', {type: 'float32', dimensions: [1, 3, 2, 2]});
    const op12Data = new Float32Array([1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0]);
    const param15 = 0;
    const param16 = 0;
    const param17 = 0;
    const param18 = 0;
    const param19 = 1;
    const param20 = 1;
    const param21 = 2;
    const param22 = 2;
    const layout = 'nchw';
    const expected = [6.442049503326416, 7.314369201660156, 8.215838432312012];
    const interOut0 = builder.l2Pool2d(op12, {'padding': [param17, param18, param15, param16], 'strides': [param20, param19], 'windowDimensions': [param22, param21], 'layout': layout});
    const op42 = builder.clamp(interOut0);
    const graph = builder.build({op42});
    const outputs = {op42: new Float32Array(utils.sizeOfShape([1, 3, 1, 1]))};
    graph.compute({'op12': op12Data}, outputs);
    utils.checkValue(outputs.op42, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
