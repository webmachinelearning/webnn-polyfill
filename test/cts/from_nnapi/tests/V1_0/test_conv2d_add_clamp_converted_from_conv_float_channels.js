'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test conv2d + add + clamp converted from conv_float_channels test', async function() {
    // Converted test case (from: V1_0/conv_float_channels.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 1, 1, 3]});
    const op1Data = new Float32Array([99.0, 99.0, 99.0]);
    const op2 = builder.constant({type: 'float32', dimensions: [3, 1, 1, 3]}, new Float32Array([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0]));
    const op3 = builder.constant({type: 'float32', dimensions: [3]}, new Float32Array([0.0, 0.0, 0.0]));
    const pad0 = 0;
    const stride = 1;
    const expected = [297.0, 594.0, 891.0];
    const interOut0 = builder.conv2d(op1, op2, {'padding': [pad0, pad0, pad0, pad0], 'strides': [stride, stride], 'inputLayout': 'nhwc', 'filterLayout': 'ohwi'});
    const interOut1 = builder.add(interOut0, op3);
    const op4 = builder.clamp(interOut1);
    const graph = await builder.build({op4});
    const outputs = await graph.compute({'op1': {data: op1Data}});
    utils.checkValue(outputs.op4.data, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
