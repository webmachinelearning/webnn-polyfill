'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test sub + clamp converted from sub_broadcast_float test', async function() {
    // Converted test case (from: V1_1/sub_broadcast_float.mod.py)
    const builder = nn.createModelBuilder();
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2]});
    const op1Buffer = new Float32Array([1, 2]);
    const op2 = builder.input('op2', {type: 'float32', dimensions: [2, 2]});
    const op2Buffer = new Float32Array([1, 2, 3, 4]);
    const expected = [0, 0, -2, -2];
    const interOut0 = builder.sub(op1, op2);
    const op3 = builder.clamp(interOut0);
    const model = builder.createModel({op3});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}, 'op2': {buffer: op2Buffer}});
    utils.checkValue(outputs.op3.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
