'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test softmax converted from softmax_float_2 test', async function() {
    // Converted test case (from: V1_0/softmax_float_2.mod.py)
    const builder = nn.createModelBuilder();
    const input = builder.input('input', {type: 'float32', dimensions: [2, 5]});
    const inputBuffer = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, -4.0, -5.0]);
    const expected = [0.011656231, 0.031684921, 0.086128544, 0.234121657, 0.636408647, 0.636408647, 0.234121657, 0.086128544, 0.031684921, 0.011656231];
    const output = builder.softmax(input);
    const model = builder.createModel({output});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input': {buffer: inputBuffer}});
    utils.checkValue(outputs.output.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
