'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('test floor converted from floor test', async () => {
    // Converted test case (from: V1_0/floor.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {dataType: 'float32', dimensions: [1, 2, 2, 2]});
    const op1Data = new Float32Array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 10.2]);
    const expected = [-2.0, -1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 10];
    const op2 = builder.floor(op1);
    const graph = await builder.build({op2});
    const outputs = {op2: new Float32Array(utils.sizeOfShape([1, 2, 2, 2]))};
    const computeResult = await context.compute(graph, {'op1': op1Data}, outputs);
    utils.checkValue(computeResult.outputs.op2, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
