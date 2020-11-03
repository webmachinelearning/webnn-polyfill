'use strict';
import * as utils from '../utils.js';

describe('test concat', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  async function testConcat(tensors, expected) {
    const builder = nn.createModelBuilder();
    const constants = [];
    for (const t of tensors) {
      constants.push(builder.constant(t.desc, new Float32Array(t.value)));
    }
    const output = builder.concat(constants, expected.axis);
    const model = builder.createModel({output});
    const compiledModel = await model.compile();
    const outputs = await compiledModel.compute();
    utils.checkShape(outputs.output.dimensions, expected.shape);
    utils.checkValue(outputs.output.buffer, expected.value);
  }

  it('concat 1d', async function() {
    const tensors = [
      {desc: {type: 'float32', dimensions: [2]}, value: [1, 2]},
      {desc: {type: 'float32', dimensions: [2]}, value: [3, 4]},
    ];
    const expected = {axis: 0, shape: [4], value: [1, 2, 3, 4]};
    await testConcat(tensors, expected);
  });

  it('concat 2d', async function() {
    const tensors = [
      {desc: {type: 'float32', dimensions: [2, 2]}, value: [1, 2, 3, 4]},
      {desc: {type: 'float32', dimensions: [2, 2]}, value: [5, 6, 7, 8]},
    ];
    const expected = [
      {axis: 0, shape: [4, 2], value: [1, 2, 3, 4, 5, 6, 7, 8]},
      {axis: 1, shape: [2, 4], value: [1, 2, 5, 6, 3, 4, 7, 8]},
    ];
    for (const test of expected) {
      await testConcat(tensors, test);
    }
  });

  it('concat 3d', async function() {
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
      await testConcat(tensors, test);
    }
  });
});
