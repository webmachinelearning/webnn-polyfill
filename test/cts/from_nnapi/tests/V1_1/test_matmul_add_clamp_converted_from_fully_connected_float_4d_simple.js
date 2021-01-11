'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test matmul + add + clamp converted from fully_connected_float_4d_simple test', async function() {
    // Converted test case (from: V1_1/fully_connected_float_4d_simple.mod.py)
    const builder = nn.createModelBuilder();
    const op1 = builder.input('op1', {type: 'float32', dimensions: [2, 10]});
    const op1Buffer = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, -9, -10, 1, 2, 3, 4, 5, 6, 7, -8, 9, -10]);
    const op2 = builder.constant({type: 'float32', dimensions: [10, 3]}, new Float32Array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10]));
    const b0 = builder.constant({type: 'float32', dimensions: [3]}, new Float32Array([1, 2, 3]));
    const expected = [24, 25, 26, 58, 59, 60];
    const interOut0 = builder.matmul(op1, op2);
    const interOut1 = builder.add(interOut0, b0);
    const op3 = builder.clamp(interOut1);
    const model = builder.createModel({op3});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}});
    utils.checkValue(outputs.op3.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
