'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('test pad converted from pad_v2_1_float test', async () => {
    // Converted test case (from: V1_2/pad_v2_1_float.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {dataType: 'float32', dimensions: [1, 2, 3, 1]});
    const input0Data = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    const beginningPadding = [0, 0, 1, 0];
    const endingPadding = [0, 2, 3, 0];
    const padValue = 9.3;
    const expected = [9.3, 1.0, 2.0, 3.0, 9.3, 9.3, 9.3, 9.3, 4.0, 5.0, 6.0, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3];
    const output0 = builder.pad(input0, beginningPadding, endingPadding, {'value': padValue});
    const graph = await builder.build({output0});
    const outputs = {output0: new Float32Array(utils.sizeOfShape([1, 4, 7, 1]))};
    const computeResult = await context.compute(graph, {'input0': input0Data}, outputs);
    utils.checkValue(computeResult.outputs.output0, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test pad converted from pad_v2_1_float_relaxed test', async () => {
    // Converted test case (from: V1_2/pad_v2_1_float.mod.py)
    const builder = new MLGraphBuilder(context);
    const input0 = builder.input('input0', {dataType: 'float32', dimensions: [1, 2, 3, 1]});
    const input0Data = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    const beginningPadding = [0, 0, 1, 0];
    const endingPadding = [0, 2, 3, 0];
    const padValue = 9.3;
    const expected = [9.3, 1.0, 2.0, 3.0, 9.3, 9.3, 9.3, 9.3, 4.0, 5.0, 6.0, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3, 9.3];
    const output0 = builder.pad(input0, beginningPadding, endingPadding, {'value': padValue});
    const graph = await builder.build({output0});
    const outputs = {output0: new Float32Array(utils.sizeOfShape([1, 4, 7, 1]))};
    const computeResult = await context.compute(graph, {'input0': input0Data}, outputs);
    utils.checkValue(computeResult.outputs.output0, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
