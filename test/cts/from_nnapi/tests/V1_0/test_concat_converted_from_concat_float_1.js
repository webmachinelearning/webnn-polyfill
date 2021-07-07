'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', function() {
  const context = navigator.ml.createContext();

  it('test concat converted from concat_float_1 test', function() {
    // Converted test case (from: V1_0/concat_float_1.mod.py)
    const builder = new MLGraphBuilder(context);
    const op1 = builder.input('op1', {type: 'float32', dimensions: [2, 3]});
    const op1Data = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    const op2 = builder.input('op2', {type: 'float32', dimensions: [2, 3]});
    const op2Data = new Float32Array([7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    const axis0 = 0;
    const expected = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    const result = builder.concat([op1, op2], axis0);
    const graph = builder.build({result});
    const outputs = {result: new Float32Array(utils.sizeOfShape([4, 3]))};
    graph.compute({'op1': op1Data, 'op2': op2Data}, outputs);
    utils.checkValue(outputs.result, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
