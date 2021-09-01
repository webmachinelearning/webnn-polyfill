'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test l2Pool2d + clamp converted from l2_pool_float_large test', function() {
    // Converted test case (from: V1_0/l2_pool_float_large.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 2, 3]});
    const op1Data = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    const pad0 = 0;
    const strideWidth = 1;
    const strideHeight = 1;
    const filterWidth = 2;
    const filterHeight = 2;
    const expected = [6.442049503326416, 7.314369201660156, 8.215838432312012];
    const interOut0 = builder.l2Pool2d(op1, {'padding': [pad0, pad0, pad0, pad0], 'strides': [strideHeight, strideWidth], 'windowDimensions': [filterHeight, filterWidth], 'layout': 'nhwc'});
    const op3 = builder.clamp(interOut0);
    const graph = builder.build({op3});
    const outputs = {op3: new Float32Array(utils.sizeOfShape([1, 1, 1, 3]))};
    graph.compute({'op1': op1Data}, outputs);
    utils.checkValue(outputs.op3, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
