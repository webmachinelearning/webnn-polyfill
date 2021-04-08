'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test add + clamp converted from add_relaxed test', async function() {
    // Converted test case (from: V1_1/add_relaxed.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [2]});
    const op1Data = new Float32Array([1.0, 2.0]);
    const op2 = builder.input('op2', {type: 'float32', dimensions: [2]});
    const op2Data = new Float32Array([3.0, 4.0]);
    const expected = [4.0, 6.0];
    const interOut0 = builder.add(op1, op2);
    const op3 = builder.clamp(interOut0);
    const graph = await builder.build({op3});
    const outputs = await graph.compute({'op1': {data: op1Data}, 'op2': {data: op2Data}});
    utils.checkValue(outputs.op3.data, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
