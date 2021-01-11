'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test transpose converted from transpose test', async function() {
    // Converted test case (from: V1_1/transpose.mod.py)
    const builder = nn.createModelBuilder();
    const input = builder.input('input', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const inputBuffer = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const perms = [0, 2, 1, 3];
    const expected = [1.0, 3.0, 2.0, 4.0];
    const output = builder.transpose(input, {'permutation': perms});
    const model = builder.createModel({output});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input': {buffer: inputBuffer}});
    utils.checkValue(outputs.output.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
