'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test div + clamp converted from div_broadcast_float_relaxed test', function() {
    // Converted test case (from: V1_1/div_broadcast_float_relaxed.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2]});
    const op1Data = new Float32Array([1, 2]);
    const op2 = builder.input('op2', {type: 'float32', dimensions: [2, 2]});
    const op2Data = new Float32Array([1, 1, 2, 2]);
    const expected = [1, 2, 0.5, 1];
    const interOut0 = builder.div(op1, op2);
    const op3 = builder.clamp(interOut0);
    const graph = builder.build({op3});
    const outputs = {op3: new Float32Array(utils.sizeOfShape([2, 2]))};
    graph.compute({'op1': op1Data, 'op2': op2Data}, outputs);
    utils.checkValue(outputs.op3, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
