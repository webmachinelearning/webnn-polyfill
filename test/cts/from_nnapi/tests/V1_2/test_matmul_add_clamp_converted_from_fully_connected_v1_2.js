'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test matmul + add + clamp converted from fully_connected_v1_2 test', function() {
    // Converted test case (from: V1_2/fully_connected_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [3, 1]});
    const op1Data = new Float32Array([2, 32, 16]);
    const op2 = builder.constant({type: 'float32', dimensions: [1, 1]}, new Float32Array([2]));
    const b0 = builder.constant({type: 'float32', dimensions: [1]}, new Float32Array([4]));
    const expected = [8, 68, 36];
    const interOut0 = builder.matmul(op1, op2);
    const interOut1 = builder.add(interOut0, b0);
    const op3 = builder.clamp(interOut1);
    const graph = builder.build({op3});
    const outputs = {op3: new Float32Array(utils.sizeOfShape([3, 1]))};
    graph.compute({'op1': op1Data}, outputs);
    utils.checkValue(outputs.op3, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test matmul + add + clamp converted from fully_connected_v1_2_relaxed test', function() {
    // Converted test case (from: V1_2/fully_connected_v1_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [3, 1]});
    const op1Data = new Float32Array([2, 32, 16]);
    const op2 = builder.constant({type: 'float32', dimensions: [1, 1]}, new Float32Array([2]));
    const b0 = builder.constant({type: 'float32', dimensions: [1]}, new Float32Array([4]));
    const expected = [8, 68, 36];
    const interOut0 = builder.matmul(op1, op2);
    const interOut1 = builder.add(interOut0, b0);
    const op3 = builder.clamp(interOut1);
    const graph = builder.build({op3});
    const outputs = {op3: new Float32Array(utils.sizeOfShape([3, 1]))};
    graph.compute({'op1': op1Data}, outputs);
    utils.checkValue(outputs.op3, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
