'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('test split converted from split_float_3 test', async () => {
    // Converted test case (from: V1_2/split_float_3.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {dataType: 'float32', dimensions: [2, 3]});
    const input0Data = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    const axis = 1;
    const numSplits = 3;
    const expected = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
    const [output0, output1, output2] = builder.split(input0, numSplits, {'axis': axis});
    const graph = await builder.build({output0, output1, output2});
    const outputs = {output0: new Float32Array(utils.sizeOfShape([2, 1])), output1: new Float32Array(utils.sizeOfShape([2, 1])), output2: new Float32Array(utils.sizeOfShape([2, 1]))};
    const computeResult = await context.compute(graph, {'input0': input0Data}, outputs);
    for (let i = 0; i < 3; i++) {
      utils.checkValue(computeResult.outputs[['output0', 'output1', 'output2'][i]], expected[i], utils.ctsFp32RestrictAccuracyCriteria);
    }
  });

  it('test split converted from split_float_3_relaxed test', async () => {
    // Converted test case (from: V1_2/split_float_3.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {dataType: 'float32', dimensions: [2, 3]});
    const input0Data = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    const axis = 1;
    const numSplits = 3;
    const expected = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
    const [output0, output1, output2] = builder.split(input0, numSplits, {'axis': axis});
    const graph = await builder.build({output0, output1, output2});
    const outputs = {output0: new Float32Array(utils.sizeOfShape([2, 1])), output1: new Float32Array(utils.sizeOfShape([2, 1])), output2: new Float32Array(utils.sizeOfShape([2, 1]))};
    const computeResult = await context.compute(graph, {'input0': input0Data}, outputs);
    for (let i = 0; i < 3; i++) {
      utils.checkValue(computeResult.outputs[['output0', 'output1', 'output2'][i]], expected[i], utils.ctsFp32RelaxedAccuracyCriteria);
    }
  });
});
/* eslint-disable max-len */
