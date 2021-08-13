'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test pad converted from pad_float_1 test', function() {
    // Converted test case (from: V1_1/pad_float_1.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [1, 2, 3, 1]});
    const op1Data = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    const op2 = builder.constant({type: 'int32', dimensions: [4, 2]}, new Int32Array([0, 0, 0, 2, 1, 3, 0, 0]));
    const expected = [0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    const op3 = builder.pad(op1, op2);
    const graph = builder.build({op3});
    const outputs = {op3: new Float32Array(utils.sizeOfShape([1, 4, 7, 1]))};
    graph.compute({'op1': op1Data}, outputs);
    utils.checkValue(outputs.op3, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
