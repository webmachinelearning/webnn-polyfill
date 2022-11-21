'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('test sigmoid converted from logistic_float_1_relaxed test', async () => {
    // Converted test case (from: V1_1/logistic_float_1_relaxed.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const op1Data = new Float32Array([1.0, 2.0, 4.0, 8.0]);
    const expected = [0.7310585975646973, 0.8807970285415649, 0.9820137619972229, 0.9996646642684937];
    const op3 = builder.sigmoid(op1);
    const graph = await builder.build({op3});
    const outputs = {op3: new Float32Array(utils.sizeOfShape([1, 2, 2, 1]))};
    await context.compute(graph, {'op1': op1Data}, outputs);
    utils.checkValue(outputs.op3, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
