'use strict';
import * as utils from '../utils.js';

describe('test concat', function() {
  const context = navigator.ml.createContext();

  function testConcatInputs(tensors, expected) {
    const builder = new MLGraphBuilder(context);
    const inputs = [];
    const namedInputs = {};
    for (let i = 0; i < tensors.length; i++) {
      inputs.push(builder.input('input' + i, tensors[i].desc));
      namedInputs['input' + i] = new Float32Array(tensors[i].value);
    }
    const output = builder.concat(inputs, expected.axis);
    const graph = builder.build({output});
    const outputs = {
      'output': new Float32Array(utils.sizeOfShape(expected.shape)),
    };
    graph.compute(namedInputs, outputs);
    utils.checkValue(outputs.output, expected.value);
  }

  it('concat 1d', function() {
    const tensors = [
      {desc: {type: 'float32', dimensions: [2]}, value: [1, 2]},
      {desc: {type: 'float32', dimensions: [2]}, value: [3, 4]},
    ];
    const expected = {axis: 0, shape: [4], value: [1, 2, 3, 4]};
    testConcatInputs(tensors, expected);
  });

  it('concat 2d', function() {
    const tensors = [
      {desc: {type: 'float32', dimensions: [2, 2]}, value: [1, 2, 3, 4]},
      {desc: {type: 'float32', dimensions: [2, 2]}, value: [5, 6, 7, 8]},
    ];
    const expected = [
      {axis: 0, shape: [4, 2], value: [1, 2, 3, 4, 5, 6, 7, 8]},
      {axis: 1, shape: [2, 4], value: [1, 2, 5, 6, 3, 4, 7, 8]},
    ];
    for (const test of expected) {
      testConcatInputs(tensors, test);
    }
  });

  it('concat 3d', function() {
    const tensors = [
      {
        desc: {type: 'float32', dimensions: [2, 2, 2]},
        value: [1, 2, 3, 4, 5, 6, 7, 8],
      },
      {
        desc: {type: 'float32', dimensions: [2, 2, 2]},
        value: [9, 10, 11, 12, 13, 14, 15, 16],
      },
    ];
    const expected = [
      {
        axis: 0,
        shape: [4, 2, 2],
        value: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
      },
      {
        axis: 1,
        shape: [2, 4, 2],
        value: [1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8, 13, 14, 15, 16],
      },
      {
        axis: 2,
        shape: [2, 2, 4],
        value: [1, 2, 9, 10, 3, 4, 11, 12, 5, 6, 13, 14, 7, 8, 15, 16],
      },
    ];
    for (const test of expected) {
      testConcatInputs(tensors, test);
    }
  });
});
