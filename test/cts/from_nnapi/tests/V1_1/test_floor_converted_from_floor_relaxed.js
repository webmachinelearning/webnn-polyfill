'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test floor converted from floor_relaxed test', function() {
    // Converted test case (from: V1_1/floor_relaxed.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 2]});
    const op1Data = new Float32Array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 10.2]);
    const expected = [-2.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 10];
    const op2 = builder.floor(op1);
    const graph = builder.build({op2});
    const outputs = {op2: new Float32Array(utils.sizeOfShape([1, 2, 2, 2]))};
    graph.compute({'op1': op1Data}, outputs);
    utils.checkValue(outputs.op2, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
