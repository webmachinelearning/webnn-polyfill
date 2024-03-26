'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('test averagePool2d + clamp converted from avg_pool_float_1 test', async () => {
    // Converted test case (from: V1_0/avg_pool_float_1.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {dataType: 'float32', dimensions: [1, 2, 2, 1]});
    const op1Data = new Float32Array([1.0, 2.0, 3.0, 4.0]);
    const pad0 = 0;
    const cons1 = 1;
    const expected = [1.0, 2.0, 3.0, 4.0];
    const interOut0 = builder.averagePool2d(op1, {'padding': [pad0, pad0, pad0, pad0], 'strides': [cons1, cons1], 'windowDimensions': [cons1, cons1], 'layout': 'nhwc'});
    const op3 = builder.clamp(interOut0);
    const graph = await builder.build({op3});
    const outputs = {op3: new Float32Array(utils.sizeOfShape([1, 2, 2, 1]))};
    const computeResult = await context.compute(graph, {'op1': op1Data}, outputs);
    utils.checkValue(computeResult.outputs.op3, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
