'use strict';
import * as utils from '../utils.js';

describe('test reduce', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  async function testReduce(op, options, input, expected) {
    const builder = nn.createModelBuilder();
    const x = builder.input('x', {type: 'float32', dimensions: input.shape});
    const y = builder['reduce' + op](x, options);
    const model = builder.createModel({y});
    const compiledModel = await model.compile();
    const inputs = {'x': {buffer: new Float32Array(input.values)}};
    const outputs = await compiledModel.compute(inputs);
    utils.checkShape(outputs.y.dimensions, expected.shape);
    utils.checkValue(outputs.y.buffer, expected.values);
  }

  it('reduceMean default axes keep dims', async function() {
    await testReduce(
        'Mean', {keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.],
        },
        {shape: [1, 1, 1], values: [18.25]});
  });

  it('reduceMean do not keep dims', async function() {
    await testReduce(
        'Mean', {axes: [1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.],
        },
        {shape: [3, 2], values: [12.5, 1.5, 35., 1.5, 57.5, 1.5]});
  });

  it('reduceMean keep dims', async function() {
    await testReduce(
        'Mean', {axes: [1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.],
        },
        {shape: [3, 1, 2], values: [12.5, 1.5, 35., 1.5, 57.5, 1.5]});
  });

  it('reduceMean negative axes keep dims', async function() {
    await testReduce(
        'Mean', {axes: [-1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [5., 1., 20., 2., 30., 1., 40., 2., 55., 1., 60., 2.],
        },
        {shape: [3, 2, 1], values: [3., 11., 15.5, 21., 28., 31.]});
  });
});
