'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test squeeze converted from squeeze_float_1 test', function() {
    // Converted test case (from: V1_1/squeeze_float_1.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {type: 'float32', dimensions: [1, 24, 1]});
    const inputData = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]);
    const squeezeDims = [2];
    const expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24];
    const output = builder.squeeze(input, {'axes': squeezeDims});
    const graph = builder.build({output});
    const outputs = {output: new Float32Array(utils.sizeOfShape([1, 24]))};
    graph.compute({'input': inputData}, outputs);
    utils.checkValue(outputs.output, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
