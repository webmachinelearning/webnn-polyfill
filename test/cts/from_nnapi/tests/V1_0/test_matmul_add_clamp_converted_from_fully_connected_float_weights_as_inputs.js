'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test matmul + add + clamp converted from fully_connected_float_weights_as_inputs test', function() {
    // Converted test case (from: V1_0/fully_connected_float_weights_as_inputs.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [3, 1]});
    const op1Data = new Float32Array([2, 32, 16]);
    const op2 = builder.input('op2', {type: 'float32', dimensions: [1, 1]});
    const op2Data = new Float32Array([2]);
    const b0 = builder.input('b0', {type: 'float32', dimensions: [1]});
    const b0Data = new Float32Array([4]);
    const expected = [8, 68, 36];
    const interOut0 = builder.matmul(op1, op2);
    const interOut1 = builder.add(interOut0, b0);
    const op3 = builder.clamp(interOut1);
    const graph = builder.build({op3});
    const outputs = {op3: new Float32Array(utils.sizeOfShape([3, 1]))};
    graph.compute({'op1': op1Data, 'op2': op2Data, 'b0': b0Data}, outputs);
    utils.checkValue(outputs.op3, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
