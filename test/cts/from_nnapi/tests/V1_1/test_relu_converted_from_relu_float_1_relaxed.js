'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test relu converted from relu_float_1_relaxed test', async function() {
    // Converted test case (from: V1_1/relu_float_1_relaxed.mod.py)
    const builder = nn.createModelBuilder();
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const op1Buffer = new Float32Array([-10.0, -0.5, 0.5, 10.0]);
    const expected = [0.0, 0.0, 0.5, 10.0];
    const op2 = builder.relu(op1);
    const model = builder.createModel({op2});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}});
    utils.checkValue(outputs.op2.buffer, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
