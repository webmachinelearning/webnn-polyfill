'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('test matmul + add + clamp converted from fully_connected_float_4d_simple test', async () => {
    // Converted test case (from: V1_1/fully_connected_float_4d_simple.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [2, 10]});
    const op1Data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, -9, -10, 1, 2, 3, 4, 5, 6, 7, -8, 9, -10]);
    const op2 = builder.constant({type: 'float32', dimensions: [10, 3]}, new Float32Array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10]));
    const b0 = builder.constant({type: 'float32', dimensions: [3]}, new Float32Array([1, 2, 3]));
    const expected = [24, 25, 26, 58, 59, 60];
    const interOut0 = builder.matmul(op1, op2);
    const interOut1 = builder.add(interOut0, b0);
    const op3 = builder.clamp(interOut1);
    const graph = await builder.build({op3});
    const outputs = {op3: new Float32Array(utils.sizeOfShape([2, 3]))};
    await context.compute(graph, {'op1': op1Data}, outputs);
    utils.checkValue(outputs.op3, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
