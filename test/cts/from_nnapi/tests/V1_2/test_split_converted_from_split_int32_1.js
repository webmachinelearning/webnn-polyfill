'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test split converted from split_int32_1 test', async function() {
    // Converted test case (from: V1_2/split_int32_1.mod.py)
    const builder = nn.createModelBuilder();
    const input0 = builder.input('input0', {type: 'int32', dimensions: [6]});
    const input0Buffer = new Int32Array([1, 2, 3, 4, 5, 6]);
    const axis = 0;
    const numSplits = 3;
    const expected = [[1, 2], [3, 4], [5, 6]];
    const [output0, output1, output2] = builder.split(input0, numSplits, {'axis': axis});
    const model = builder.createModel({output0, output1, output2});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input0': {buffer: input0Buffer}});
    for (let i = 0; i < 3; i++) {
      utils.checkValue(outputs[['output0', 'output1', 'output2'][i]].buffer, expected[i], utils.ctsFp32RestrictAccuracyCriteria);
    }
  });

  it('test split converted from split_int32_1_relaxed test', async function() {
    // Converted test case (from: V1_2/split_int32_1.mod.py)
    const builder = nn.createModelBuilder();
    const input0 = builder.input('input0', {type: 'int32', dimensions: [6]});
    const input0Buffer = new Int32Array([1, 2, 3, 4, 5, 6]);
    const axis = 0;
    const numSplits = 3;
    const expected = [[1, 2], [3, 4], [5, 6]];
    const [output0, output1, output2] = builder.split(input0, numSplits, {'axis': axis});
    const model = builder.createModel({output0, output1, output2});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input0': {buffer: input0Buffer}});
    for (let i = 0; i < 3; i++) {
      utils.checkValue(outputs[['output0', 'output1', 'output2'][i]].buffer, expected[i], utils.ctsFp32RelaxedAccuracyCriteria);
    }
  });
});
/* eslint-disable max-len */
