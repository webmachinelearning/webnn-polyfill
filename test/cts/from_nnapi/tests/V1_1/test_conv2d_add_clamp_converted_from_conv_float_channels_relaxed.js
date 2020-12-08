'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test conv2d + add + clamp converted from conv_float_channels_relaxed test', async function() {
    // Converted test case (from: V1_1/conv_float_channels_relaxed.mod.py)
    const builder = nn.createModelBuilder();
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 1, 1, 3]});
    const op1Buffer = new Float32Array([99.0, 99.0, 99.0]);
    const op2 = builder.constant({type: 'float32', dimensions: [1, 1, 3, 3]}, new Float32Array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]));
    const op3 = builder.constant({type: 'float32', dimensions: [3]}, new Float32Array([0.0, 0.0, 0.0]));
    const pad0 = 0;
    const stride = 1;
    const expected = [297.0, 594.0, 891.0];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [pad0, pad0, pad0, pad0], 'strides': [stride, stride], 'layout': 'nhwc'});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const model = builder.createModel({op4});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}});
    utils.checkValue(outputs.op4.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
