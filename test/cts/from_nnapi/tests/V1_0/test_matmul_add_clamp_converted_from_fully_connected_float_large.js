'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('test matmul + add + clamp converted from fully_connected_float_large test', async () => {
    // Converted test case (from: V1_0/fully_connected_float_large.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 5]});
    const op1Data = new Float32Array([1, 10, 100, 1000, 10000]);
    const op2 = builder.constant({type: 'float32', dimensions: [5, 1]}, new Float32Array([2, 3, 4, 5, 6]));
    const b0 = builder.constant({type: 'float32', dimensions: [1]}, new Float32Array([900000]));
    const expected = [965432];
    const interOut0 = builder.matmul(op1, op2);
    const interOut1 = builder.add(interOut0, b0);
    const op3 = builder.clamp(interOut1);
    const graph = await builder.build({op3});
    const outputs = {op3: new Float32Array(utils.sizeOfShape([1, 1]))};
    await context.compute(graph, {'op1': op1Data}, outputs);
    utils.checkValue(outputs.op3, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
