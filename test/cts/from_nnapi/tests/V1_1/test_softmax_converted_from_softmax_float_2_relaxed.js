'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test softmax converted from softmax_float_2_relaxed test', function() {
    // Converted test case (from: V1_1/softmax_float_2_relaxed.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {type: 'float32', dimensions: [2, 5]});
    const inputData = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, -1.0, -2.0, -3.0, -4.0, -5.0]);
    const expected = [0.011656231, 0.031684921, 0.086128544, 0.234121657, 0.636408647, 0.636408647, 0.234121657, 0.086128544, 0.031684921, 0.011656231];
    const output = builder.softmax(input);
    const graph = builder.build({output});
    const outputs = {output: new Float32Array(utils.sizeOfShape([2, 5]))};
    graph.compute({'input': inputData}, outputs);
    utils.checkValue(outputs.output, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
