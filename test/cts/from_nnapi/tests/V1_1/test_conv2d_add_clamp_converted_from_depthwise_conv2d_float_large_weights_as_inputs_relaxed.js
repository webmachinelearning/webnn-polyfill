'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test conv2d + add + clamp converted from depthwise_conv2d_float_large_weights_as_inputs_relaxed test', async function() {
    // Converted test case (from: V1_1/depthwise_conv2d_float_large_weights_as_inputs_relaxed.mod.py)
    const builder = nn.createModelBuilder();
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 2]});
    const op1Buffer = new Float32Array([10, 21, 10, 22, 10, 23, 10, 24]);
    const op2 = builder.input('op2', {type: 'float32', dimensions: [2, 2, 1, 2]});
    const op2Buffer = new Float32Array([0.25, 0.0, 0.25, 1.0, 0.25, 0.0, 0.25, 1.0]);
    const op3 = builder.input('op3', {type: 'float32', dimensions: [2]});
    const op3Buffer = new Float32Array([100, 200]);
    const pad0 = 0;
    const stride = 1;
    const expected = [110, 246];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [pad0, pad0, pad0, pad0], 'strides': [stride, stride], 'inputLayout': 'nhwc', 'filterLayout': 'hwio', 'groups': 2});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const model = builder.createModel({op4});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}, 'op2': {buffer: op2Buffer}, 'op3': {buffer: op3Buffer}});
    utils.checkValue(outputs.op4.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
