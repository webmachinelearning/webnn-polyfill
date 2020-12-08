'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test add + clamp converted from add_relaxed test', async function() {
    // Converted test case (from: V1_1/add_relaxed.mod.py)
    const builder = nn.createModelBuilder();
    const op1 = builder.input('op1', {type: 'float32', dimensions: [2]});
    const op1Buffer = new Float32Array([1.0, 2.0]);
    const op2 = builder.input('op2', {type: 'float32', dimensions: [2]});
    const op2Buffer = new Float32Array([3.0, 4.0]);
    const expected = [4.0, 6.0];
    const interOut0 = builder.add(op1, op2);
    const op3 = builder.clamp(interOut0);
    const model = builder.createModel({op3});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}, 'op2': {buffer: op2Buffer}});
    utils.checkValue(outputs.op3.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
