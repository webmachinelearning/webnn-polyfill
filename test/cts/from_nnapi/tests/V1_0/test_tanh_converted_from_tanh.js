'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test tanh converted from tanh test', async function() {
    // Converted test case (from: V1_0/tanh.mod.py)
    const builder = nn.createModelBuilder();
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const op1Buffer = new Float32Array([-1, 0, 1, 10]);
    const expected = [-0.761594156, 0, 0.761594156, 0.999999996];
    const op2 = builder.tanh(op1);
    const model = builder.createModel({op2});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}});
    utils.checkValue(outputs.op2.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
