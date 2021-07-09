'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test conv2d + add + clamp converted from depthwise_conv2d_float_large test', function() {
    // Converted test case (from: V1_0/depthwise_conv2d_float_large.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 2]});
    const op1Data = new Float32Array([10, 21, 10, 22, 10, 23, 10, 24]);
    const op2 = builder.constant({type: 'float32', dimensions: [1, 2, 2, 2]}, new Float32Array([0.25, 0, 0.25, 1, 0.25, 0, 0.25, 1]));
    const op3 = builder.constant({type: 'float32', dimensions: [2]}, new Float32Array([100, 200]));
    const pad0 = 0;
    const stride = 1;
    const expected = [110, 246];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [pad0, pad0, pad0, pad0], 'strides': [stride, stride], 'inputLayout': 'nhwc', 'groups': 2, 'filterLayout': 'ihwo'});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const graph = builder.build({op4});
    const outputs = {op4: new Float32Array(utils.sizeOfShape([1, 1, 1, 2]))};
    graph.compute({'op1': op1Data}, outputs);
    utils.checkValue(outputs.op4, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
