'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('test conv2d (fused ops) converted from depthwise_conv2d_float_large_2 test', async () => {
    // Converted test case (from: V1_0/depthwise_conv2d_float_large_2.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 4]});
    const op1Data = new Float32Array([10, 21, 100, 0, 10, 22, 200, 0, 10, 23, 300, 0, 10, 24, 400, 0]);
    const op2 = builder.constant({type: 'float32', dimensions: [1, 2, 2, 4]}, new Float32Array([0.25, 0, 10, 100, 0.25, 1, 20, 100, 0.25, 0, 30, 100, 0.25, 1, 40, 100]));
    const op3 = builder.constant({type: 'float32', dimensions: [4]}, new Float32Array([600000, 700000, 800000, 900000]));
    const pad0 = 0;
    const stride = 1;
    const expected = [600010, 700046, 830000, 900000];
    const op4 = builder.conv2d(op1, op2, {'bias': op3, 'padding': [pad0, pad0, pad0, pad0], 'strides': [stride, stride], 'inputLayout': 'nhwc', 'groups': 4, 'filterLayout': 'ihwo'});
    const graph = await builder.build({op4});
    const outputs = {op4: new Float32Array(utils.sizeOfShape([1, 1, 1, 4]))};
    const computeResult = await context.compute(graph, {'op1': op1Data}, outputs);
    utils.checkValue(computeResult.outputs.op4, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
