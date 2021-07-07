'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test transpose converted from transpose_v1_2 test', function() {
    // Converted test case (from: V1_2/transpose_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {type: 'float32', dimensions: [2, 2]});
    const inputData = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const expected = [1.0, 3.0, 2.0, 4.0];
    const output = builder.transpose(input);
    const graph = builder.build({output});
    const outputs = {output: new Float32Array(utils.sizeOfShape([2, 2]))};
    graph.compute({'input': inputData}, outputs);
    utils.checkValue(outputs.output, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test transpose converted from transpose_v1_2_relaxed test', function() {
    // Converted test case (from: V1_2/transpose_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {type: 'float32', dimensions: [2, 2]});
    const inputData = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const expected = [1.0, 3.0, 2.0, 4.0];
    const output = builder.transpose(input);
    const graph = builder.build({output});
    const outputs = {output: new Float32Array(utils.sizeOfShape([2, 2]))};
    graph.compute({'input': inputData}, outputs);
    utils.checkValue(outputs.output, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
