'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test softmax converted from softmax_v1_2_axis_dim2_axis1 test', function() {
    // Converted test case (from: V1_2/softmax_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [2, 5]});
    const op1Data = new Float32Array([17.0, 16.0, 15.0, 14.0, 1.0, -1.0, -2.0, -3.0, -4.0, -17.0]);
    const expected = [0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08, 0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08];
    const op2 = builder.softmax(op1);
    const graph = builder.build({op2});
    const outputs = {op2: new Float32Array(utils.sizeOfShape([2, 5]))};
    graph.compute({'op1': op1Data}, outputs);
    utils.checkValue(outputs.op2, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test softmax converted from softmax_v1_2_axis_dim2_axis1_neg test', function() {
    // Converted test case (from: V1_2/softmax_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [2, 5]});
    const op1Data = new Float32Array([17.0, 16.0, 15.0, 14.0, 1.0, -1.0, -2.0, -3.0, -4.0, -17.0]);
    const expected = [0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08, 0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08];
    const op2 = builder.softmax(op1);
    const graph = builder.build({op2});
    const outputs = {op2: new Float32Array(utils.sizeOfShape([2, 5]))};
    graph.compute({'op1': op1Data}, outputs);
    utils.checkValue(outputs.op2, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test softmax converted from softmax_v1_2_axis_relaxed_dim2_axis1 test', function() {
    // Converted test case (from: V1_2/softmax_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [2, 5]});
    const op1Data = new Float32Array([17.0, 16.0, 15.0, 14.0, 1.0, -1.0, -2.0, -3.0, -4.0, -17.0]);
    const expected = [0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08, 0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08];
    const op2 = builder.softmax(op1);
    const graph = builder.build({op2});
    const outputs = {op2: new Float32Array(utils.sizeOfShape([2, 5]))};
    graph.compute({'op1': op1Data}, outputs);
    utils.checkValue(outputs.op2, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test softmax converted from softmax_v1_2_axis_relaxed_dim2_axis1_neg test', function() {
    // Converted test case (from: V1_2/softmax_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [2, 5]});
    const op1Data = new Float32Array([17.0, 16.0, 15.0, 14.0, 1.0, -1.0, -2.0, -3.0, -4.0, -17.0]);
    const expected = [0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08, 0.643914213228014, 0.236882800924671, 0.087144312427294, 0.032058600957022, 7.246299848982885e-08];
    const op2 = builder.softmax(op1);
    const graph = builder.build({op2});
    const outputs = {op2: new Float32Array(utils.sizeOfShape([2, 5]))};
    graph.compute({'op1': op1Data}, outputs);
    utils.checkValue(outputs.op2, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
