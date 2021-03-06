'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test matmul + add + clamp converted from fully_connected_float_weights_as_inputs test', async function() {
    // Converted test case (from: V1_0/fully_connected_float_weights_as_inputs.mod.py)
    const builder = nn.createModelBuilder();
    const op1 = builder.input('op1', {type: 'float32', dimensions: [3, 1]});
    const op1Buffer = new Float32Array([2, 32, 16]);
    const op2 = builder.input('op2', {type: 'float32', dimensions: [1, 1]});
    const op2Buffer = new Float32Array([2]);
    const b0 = builder.input('b0', {type: 'float32', dimensions: [1]});
    const b0Buffer = new Float32Array([4]);
    const expected = [8, 68, 36];
    const interOut0 = builder.matmul(op1, op2);
    const interOut1 = builder.add(interOut0, b0);
    const op3 = builder.clamp(interOut1);
    const model = builder.createModel({op3});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'op1': {buffer: op1Buffer}, 'op2': {buffer: op2Buffer}, 'b0': {buffer: b0Buffer}});
    utils.checkValue(outputs.op3.buffer, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
