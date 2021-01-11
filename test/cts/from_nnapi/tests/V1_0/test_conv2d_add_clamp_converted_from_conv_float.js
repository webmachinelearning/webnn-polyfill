'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test conv2d + add + clamp converted from conv_float test', async function() {
    // Converted test case (from: V1_0/conv_float.mod.py)
    const builder = nn.createModelBuilder();
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 3, 3, 1]});
    const op1Buffer = new Float32Array([1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0]);
    const op2 = builder.constant({type: 'float32', dimensions: [2, 2, 1, 1]}, new Float32Array([0.25, 0.25, 0.25, 0.25]));
    const op3 = builder.constant({type: 'float32', dimensions: [1]}, new Float32Array([0]));
    const pad0 = 0;
    const stride = 1;
    const expected = [0.875, 0.875, 0.875, 0.875];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [pad0, pad0, pad0, pad0], 'strides': [stride, stride], 'layout': 'nhwc'});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const model = builder.createModel({op4});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}});
    utils.checkValue(outputs.op4.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
