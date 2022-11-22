'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('test transpose converted from transpose_relaxed test', async () => {
    // Converted test case (from: V1_1/transpose_relaxed.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {type: 'float32', dimensions: [1, 2, 2, 1]});
    const inputData = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const perms = [0, 2, 1, 3];
    const expected = [1.0, 3.0, 2.0, 4.0];
    const output = builder.transpose(input, {'permutation': perms});
    const graph = await builder.build({output});
    const outputs = {output: new Float32Array(utils.sizeOfShape([1, 2, 2, 1]))};
    await context.compute(graph, {'input': inputData}, outputs);
    utils.checkValue(outputs.output, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
