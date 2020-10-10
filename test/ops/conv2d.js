'use strict';
import * as utils from '../utils.js';

describe('test conv2d', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  it('conv2d with padding', async function() {
    const builder = nn.createModelBuilder();
    const input =
        builder.input('input', {type: 'float32', dimensions: [1, 1, 5, 5]});
    const filter = builder.constant(
        {type: 'float32', dimensions: [1, 1, 3, 3]},
        new Float32Array(9).fill(1));
    const padding = [1, 1, 1, 1];
    const output = builder.conv2d(input, filter, padding);
    const model = builder.createModel({'output': output});
    const compiledModel = await model.compile();
    const inputs = {
      'input': {
        buffer: new Float32Array([
          0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        ]),
      },
    };
    const outputs = await compiledModel.compute(inputs);
    utils.checkShape(outputs.output.dimensions, [1, 1, 5, 5]);
    const expected = [
      12.,  21., 27., 33.,  24.,  33.,  54.,  63., 72.,  51.,  63.,  99., 108.,
      117., 81., 93., 144., 153., 162., 111., 72., 111., 117., 123., 84.,
    ];
    utils.checkOutput(outputs.output.buffer, expected);
  });

  it('conv2d without padding', async function() {
    const builder = nn.createModelBuilder();
    const input =
        builder.input('input', {type: 'float32', dimensions: [1, 1, 5, 5]});
    const filter = builder.constant(
        {type: 'float32', dimensions: [1, 1, 3, 3]},
        new Float32Array(9).fill(1));
    const output = builder.conv2d(input, filter);
    const model = builder.createModel({'output': output});
    const compiledModel = await model.compile();
    const inputs = {
      'input': {
        buffer: new Float32Array([
          0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
          13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        ]),
      },
    };
    const outputs = await compiledModel.compute(inputs);
    utils.checkShape(outputs.output.dimensions, [1, 1, 3, 3]);
    const expected = [54., 63., 72., 99., 108., 117., 144., 153., 162.];
    utils.checkOutput(outputs.output.buffer, expected);
  });

  it('conv2d with strides=2 and padding', async function() {
    const builder = nn.createModelBuilder();
    const input =
        builder.input('input', {type: 'float32', dimensions: [1, 1, 7, 5]});
    const filter = builder.constant(
        {type: 'float32', dimensions: [1, 1, 3, 3]},
        new Float32Array(9).fill(1));
    const padding = [1, 1, 1, 1];
    const strides = [2, 2];
    const output = builder.conv2d(input, filter, padding, strides);
    const model = builder.createModel({'output': output});
    const compiledModel = await model.compile();
    const inputs = {
      'input': {
        buffer: new Float32Array([
          0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
          12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
          24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
        ]),
      },
    };
    const outputs = await compiledModel.compute(inputs);
    utils.checkShape(outputs.output.dimensions, [1, 1, 4, 3]);
    const expected =
        [12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.];
    utils.checkOutput(outputs.output.buffer, expected);
  });
});
