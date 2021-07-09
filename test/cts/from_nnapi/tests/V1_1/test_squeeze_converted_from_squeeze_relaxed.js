'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test squeeze converted from squeeze_relaxed test', function() {
    // Converted test case (from: V1_1/squeeze_relaxed.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {type: 'float32', dimensions: [4, 1, 1, 2]});
    const inputData = new Float32Array([1.4, 2.3, 3.2, 4.1, 5.4, 6.3, 7.2, 8.1]);
    const squeezeDims = [1, 2];
    const expected = [1.4, 2.3, 3.2, 4.1, 5.4, 6.3, 7.2, 8.1];
    const output = builder.squeeze(input, {'axes': squeezeDims});
    const graph = builder.build({output});
    const outputs = {output: new Float32Array(utils.sizeOfShape([4, 2]))};
    graph.compute({'input': inputData}, outputs);
    utils.checkValue(outputs.output, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
