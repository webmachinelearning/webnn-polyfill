'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('test reshape converted from reshape test', async () => {
    // Converted test case (from: V1_0/reshape.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 1, 3, 3]});
    const op1Data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9]);
    const op2 = [null];
    const expected = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    const op3 = builder.reshape(op1, op2);
    const graph = await builder.build({op3});
    const outputs = {op3: new Float32Array(utils.sizeOfShape([9]))};
    const computeResult = await context.compute(graph, {'op1': op1Data}, outputs);
    utils.checkValue(computeResult.outputs.op3, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
