'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test split converted from split_float_5 test', async function() {
    // Converted test case (from: V1_2/split_float_5.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {type: 'float32', dimensions: [2, 2, 2]});
    const input0Data = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    const axis = -2;
    const numSplits = 2;
    const expected = [[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]];
    const [output0, output1] = builder.split(input0, numSplits, {'axis': axis});
    const graph = await builder.build({output0, output1});
    const outputs = await graph.compute({'input0': {data: input0Data}});
    for (let i = 0; i < 2; i++) {
      utils.checkValue(outputs[['output0', 'output1'][i]].data, expected[i], utils.ctsFp32RestrictAccuracyCriteria);
    }
  });

  it('test split converted from split_float_5_relaxed test', async function() {
    // Converted test case (from: V1_2/split_float_5.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {type: 'float32', dimensions: [2, 2, 2]});
    const input0Data = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    const axis = -2;
    const numSplits = 2;
    const expected = [[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]];
    const [output0, output1] = builder.split(input0, numSplits, {'axis': axis});
    const graph = await builder.build({output0, output1});
    const outputs = await graph.compute({'input0': {data: input0Data}});
    for (let i = 0; i < 2; i++) {
      utils.checkValue(outputs[['output0', 'output1'][i]].data, expected[i], utils.ctsFp32RelaxedAccuracyCriteria);
    }
  });
});
/* eslint-disable max-len */
