'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test concat converted from concat_float_1 test', async function() {
    // Converted test case (from: V1_0/concat_float_1.mod.py)
    const builder = nn.createModelBuilder();
    const op1 = builder.input('op1', {type: 'float32', dimensions: [2, 3]});
    const op1Buffer = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    const op2 = builder.input('op2', {type: 'float32', dimensions: [2, 3]});
    const op2Buffer = new Float32Array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    const axis0 = 0;
    const expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    const result = builder.concat([op1, op2], axis0);
    const model = builder.createModel({result});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}, 'op2': {buffer: op2Buffer}});
    utils.checkValue(outputs.result.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
