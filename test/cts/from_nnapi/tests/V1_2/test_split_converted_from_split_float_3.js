'use strict';
import * as utils from '../../../../utils.js';

describe('CTS converted from NNAPI CTS', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('test split converted from split_float_3 test', async function() {
    // Converted test case (from: V1_2/split_float_3.mod.py)
    const builder = nn.createModelBuilder();
    const input0 = builder.input('input0', {type: 'float32', dimensions: [2, 3]});
    const input0Buffer = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    const axis = 1;
    const num_splits = 3;
    const expected = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
    const [output0, output1, output2] = builder.split(input0, num_splits, {'axis': axis});
    const model = builder.createModel({output0, output1, output2});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input0': {buffer: input0Buffer}});
    for (let i = 0; i < 3; i++) {
      utils.checkValue(outputs[['output0', 'output1', 'output2'][i]].buffer, expected[i], 1e-5, 5.0 * 1.1920928955078125e-7);
    }
  });

  it('test split converted from split_float_3_relaxed test', async function() {
    // Converted test case (from: V1_2/split_float_3.mod.py)
    const builder = nn.createModelBuilder();
    const input0 = builder.input('input0', {type: 'float32', dimensions: [2, 3]});
    const input0Buffer = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    const axis = 1;
    const num_splits = 3;
    const expected = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
    const [output0, output1, output2] = builder.split(input0, num_splits, {'axis': axis});
    const model = builder.createModel({output0, output1, output2});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input0': {buffer: input0Buffer}});
    for (let i = 0; i < 3; i++) {
      utils.checkValue(outputs[['output0', 'output1', 'output2'][i]].buffer, expected[i], 5.0 * 0.0009765625, 5.0 * 0.0009765625);
    }
  });
});