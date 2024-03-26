'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('test add + clamp converted from add test', async () => {
    // Converted test case (from: V1_0/add.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {dataType: 'float32', dimensions: [2]});
    const op1Data = new Float32Array([1.0, 2.0]);
    const op2 = builder.input('op2', {dataType: 'float32', dimensions: [2]});
    const op2Data = new Float32Array([3.0, 4.0]);
    const expected = [4.0, 6.0];
    const interOut0 = builder.add(op1, op2);
    const op3 = builder.clamp(interOut0);
    const graph = await builder.build({op3});
    const outputs = {op3: new Float32Array(utils.sizeOfShape([2]))};
    const computeResult = await context.compute(graph, {'op1': op1Data, 'op2': op2Data}, outputs);
    utils.checkValue(computeResult.outputs.op3, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
