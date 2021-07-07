'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test split converted from split_int32_4 test', function() {
    // Converted test case (from: V1_2/split_int32_4.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {type: 'int32', dimensions: [2, 2, 2]});
    const input0Data = new Int32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const axis = 1;
    const numSplits = 2;
    const expected = [[1, 2, 5, 6], [3, 4, 7, 8]];
    const [output0, output1] = builder.split(input0, numSplits, {'axis': axis});
    const graph = builder.build({output0, output1});
    const outputs = {output0: new Int32Array(utils.sizeOfShape([2, 1, 2])), output1: new Int32Array(utils.sizeOfShape([2, 1, 2]))};
    graph.compute({'input0': input0Data}, outputs);
    for (let i = 0; i < 2; i++) {
      utils.checkValue(outputs[['output0', 'output1'][i]], expected[i], utils.ctsFp32RestrictAccuracyCriteria);
    }
  });

  it('test split converted from split_int32_4_relaxed test', function() {
    // Converted test case (from: V1_2/split_int32_4.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {type: 'int32', dimensions: [2, 2, 2]});
    const input0Data = new Int32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const axis = 1;
    const numSplits = 2;
    const expected = [[1, 2, 5, 6], [3, 4, 7, 8]];
    const [output0, output1] = builder.split(input0, numSplits, {'axis': axis});
    const graph = builder.build({output0, output1});
    const outputs = {output0: new Int32Array(utils.sizeOfShape([2, 1, 2])), output1: new Int32Array(utils.sizeOfShape([2, 1, 2]))};
    graph.compute({'input0': input0Data}, outputs);
    for (let i = 0; i < 2; i++) {
      utils.checkValue(outputs[['output0', 'output1'][i]], expected[i], utils.ctsFp32RelaxedAccuracyCriteria);
    }
  });
});
/* eslint-disable max-len */
