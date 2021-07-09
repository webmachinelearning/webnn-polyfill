'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test relu converted from relu_float_1 test', function() {
    // Converted test case (from: V1_0/relu_float_1.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const op1Data = new Float32Array([-10.0, -0.5, 0.5, 10.0]);
    const expected = [0.0, 0.0, 0.5, 10.0];
    const op2 = builder.relu(op1);
    const graph = builder.build({op2});
    const outputs = {op2: new Float32Array(utils.sizeOfShape([1, 2, 2, 1]))};
    graph.compute({'op1': op1Data}, outputs);
    utils.checkValue(outputs.op2, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
