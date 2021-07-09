'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test tanh converted from tanh test', function() {
    // Converted test case (from: V1_0/tanh.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const op1Data = new Float32Array([-1, 0, 1, 10]);
    const expected = [-0.761594156, 0, 0.761594156, 0.999999996];
    const op2 = builder.tanh(op1);
    const graph = builder.build({op2});
    const outputs = {op2: new Float32Array(utils.sizeOfShape([1, 2, 2, 1]))};
    graph.compute({'op1': op1Data}, outputs);
    utils.checkValue(outputs.op2, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
