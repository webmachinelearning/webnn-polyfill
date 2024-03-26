'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('test pad converted from pad_low_rank test', async () => {
    // Converted test case (from: V1_2/pad_low_rank.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {dataType: 'float32', dimensions: [3]});
    const input0Data = new Float32Array([1.0, 2.0, 3.0]);
    const beginningPadding = [3];
    const endingPadding = [1];
    const expected = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0];
    const output0 = builder.pad(input0, beginningPadding, endingPadding);
    const graph = await builder.build({output0});
    const outputs = {output0: new Float32Array(utils.sizeOfShape([7]))};
    const computeResult = await context.compute(graph, {'input0': input0Data}, outputs);
    utils.checkValue(computeResult.outputs.output0, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
