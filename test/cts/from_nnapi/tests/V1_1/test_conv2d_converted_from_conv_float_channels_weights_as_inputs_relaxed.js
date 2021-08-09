'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test conv2d (fused ops) converted from conv_float_channels_weights_as_inputs_relaxed test', function() {
    // Converted test case (from: V1_1/conv_float_channels_weights_as_inputs_relaxed.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 1, 1, 3]});
    const op1Data = new Float32Array([99.0, 99.0, 99.0]);
    const op2 = builder.input('op2', {type: 'float32', dimensions: [3, 1, 1, 3]});
    const op2Data = new Float32Array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]);
    const op3 = builder.input('op3', {type: 'float32', dimensions: [3]});
    const op3Data = new Float32Array([0.0, 0.0, 0.0]);
    const pad0 = 0;
    const stride = 1;
    const expected = [297.0, 594.0, 891.0];
    const op4 = builder.conv2d(op1, op2, {'bias': op3, 'padding': [pad0, pad0, pad0, pad0], 'strides': [stride, stride], 'inputLayout': 'nhwc', 'filterLayout': 'ohwi'});
    const graph = builder.build({op4});
    const outputs = {op4: new Float32Array(utils.sizeOfShape([1, 1, 1, 3]))};
    graph.compute({'op1': op1Data, 'op2': op2Data, 'op3': op3Data}, outputs);
    utils.checkValue(outputs.op4, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
