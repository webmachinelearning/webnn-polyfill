'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test reshape converted from reshape test', async function() {
    // Converted test case (from: V1_0/reshape.mod.py)
    const builder = nn.createModelBuilder();
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 1, 3, 3]});
    const op1Buffer = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const op2 = [-1];
    const expected = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    const op3 = builder.reshape(op1, op2);
    const model = builder.createModel({op3});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}});
    utils.checkValue(outputs.op3.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
