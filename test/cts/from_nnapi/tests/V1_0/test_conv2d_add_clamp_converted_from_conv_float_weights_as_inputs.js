'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test conv2d + add + clamp converted from conv_float_weights_as_inputs test', async function() {
    // Converted test case (from: V1_0/conv_float_weights_as_inputs.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 3, 3, 1]});
    const op1Data = new Float32Array([1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]);
    const op2 = builder.input('op2', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const op2Data = new Float32Array([0.25, 0.25, 0.25, 0.25]);
    const op3 = builder.input('op3', {type: 'float32', dimensions: [1]});
    const op3Data = new Float32Array([0]);
    const pad0 = 0;
    const stride = 1;
    const expected = [0.875, 0.875, 0.875, 0.875];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [pad0, pad0, pad0, pad0], 'strides': [stride, stride], 'inputLayout': 'nhwc', 'filterLayout': 'ohwi'});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const graph = await builder.build({op4});
    const outputs = await graph.compute({'op1': {data: op1Data}, 'op2': {data: op2Data}, 'op3': {data: op3Data}});
    utils.checkValue(outputs.op4.data, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
