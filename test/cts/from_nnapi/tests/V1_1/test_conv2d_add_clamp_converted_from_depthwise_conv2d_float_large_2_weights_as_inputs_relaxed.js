'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test conv2d + add + clamp converted from depthwise_conv2d_float_large_2_weights_as_inputs_relaxed test', function() {
    // Converted test case (from: V1_1/depthwise_conv2d_float_large_2_weights_as_inputs_relaxed.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 4]});
    const op1Data = new Float32Array([10, 21, 10, 0, 10, 22, 20, 0, 10, 23, 30, 0, 10, 24, 40, 0]);
    const op2 = builder.input('op2', {type: 'float32', dimensions: [1, 2, 2, 4]});
    const op2Data = new Float32Array([0.25, 0, 10, 100, 0.25, 1, 20, 100, 0.25, 0, 30, 100, 0.25, 1, 40, 100]);
    const op3 = builder.input('op3', {type: 'float32', dimensions: [4]});
    const op3Data = new Float32Array([6000, 7000, 8000, 9000]);
    const pad0 = 0;
    const stride = 1;
    const expected = [6010, 7046, 11000, 9000];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [pad0, pad0, pad0, pad0], 'strides': [stride, stride], 'inputLayout': 'nhwc', 'groups': 4, 'filterLayout': 'ihwo'});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const graph = builder.build({op4});
    const outputs = {op4: new Float32Array(utils.sizeOfShape([1, 1, 1, 4]))};
    graph.compute({'op1': op1Data, 'op2': op2Data, 'op3': op3Data}, outputs);
    utils.checkValue(outputs.op4, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
