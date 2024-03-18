'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('test matmul + add + clamp converted from fully_connected_float_large_weights_as_inputs test', async () => {
    // Converted test case (from: V1_0/fully_connected_float_large_weights_as_inputs.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {dataType: 'float32', dimensions: [1, 5]});
    const op1Data = new Float32Array([1, 10, 100, 1000, 10000]);
    const op2 = builder.input('op2', {dataType: 'float32', dimensions: [5, 1]});
    const op2Data = new Float32Array([2, 3, 4, 5, 6]);
    const b0 = builder.input('b0', {dataType: 'float32', dimensions: [1]});
    const b0Data = new Float32Array([900000]);
    const expected = [965432];
    const interOut0 = builder.matmul(op1, op2);
    const interOut1 = builder.add(interOut0, b0);
    const op3 = builder.clamp(interOut1);
    const graph = await builder.build({op3});
    const outputs = {op3: new Float32Array(utils.sizeOfShape([1, 1]))};
    const computeResult = await context.compute(graph, {'op1': op1Data, 'op2': op2Data, 'b0': b0Data}, outputs);
    utils.checkValue(computeResult.outputs.op3, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
