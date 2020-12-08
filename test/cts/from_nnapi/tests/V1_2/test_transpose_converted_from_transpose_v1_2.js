'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test transpose converted from transpose_v1_2 test', async function() {
    // Converted test case (from: V1_2/transpose_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const input = builder.input('input', {type: 'float32', dimensions: [2, 2]});
    const inputBuffer = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const expected = [1.0, 3.0, 2.0, 4.0];
    const output = builder.transpose(input);
    const model = builder.createModel({output});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input': {buffer: inputBuffer}});
    utils.checkValue(outputs.output.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test transpose converted from transpose_v1_2_relaxed test', async function() {
    // Converted test case (from: V1_2/transpose_v1_2.mod.py)
    const builder = nn.createModelBuilder();
    const input = builder.input('input', {type: 'float32', dimensions: [2, 2]});
    const inputBuffer = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const expected = [1.0, 3.0, 2.0, 4.0];
    const output = builder.transpose(input);
    const model = builder.createModel({output});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input': {buffer: inputBuffer}});
    utils.checkValue(outputs.output.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
