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
    const output = builder.conv2d(input, filter, {padding});
    const model = builder.createModel({output});
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
    utils.checkValue(outputs.output.buffer, expected);
  });

  it('conv2d without padding', async function() {
    const builder = nn.createModelBuilder();
    const input =
        builder.input('input', {type: 'float32', dimensions: [1, 1, 5, 5]});
    const filter = builder.constant(
        {type: 'float32', dimensions: [1, 1, 3, 3]},
        new Float32Array(9).fill(1));
    const output = builder.conv2d(input, filter);
    const model = builder.createModel({output});
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
    utils.checkValue(outputs.output.buffer, expected);
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
    const output = builder.conv2d(input, filter, {padding, strides});
    const model = builder.createModel({output});
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
    utils.checkValue(outputs.output.buffer, expected);
  });

  it('conv2d with strides=2 and asymetric padding', async function() {
    const builder = nn.createModelBuilder();
    const input =
        builder.input('input', {type: 'float32', dimensions: [1, 1, 5, 5]});
    const filter = builder.constant(
        {type: 'float32', dimensions: [1, 1, 4, 2]},
        new Float32Array(8).fill(1));
    const padding = [1, 2, 0, 1];
    const strides = [2, 2];
    const output = builder.conv2d(input, filter, {padding, strides});
    const model = builder.createModel({output});
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
    const expected = [33, 45, 27, 104, 120, 66, 72, 80, 43];
    utils.checkValue(outputs.output.buffer, expected);
  });

  it('conv2d depthwise nhwc', async function() {
    // It is based on Android NNAPI CTS: V1_2/depthwise_conv2d_v1_2.mod.py
    const builder = nn.createModelBuilder();
    const input =
        builder.input('input', {type: 'float32', dimensions: [1, 2, 2, 4]});
    const inputBuffer = new Float32Array(
        [10, 21, 10, 0, 10, 22, 20, 0, 10, 23, 30, 0, 10, 24, 40, 0]);
    const filter = builder.constant(
        {type: 'float32', dimensions: [2, 2, 1, 4]}, new Float32Array([
          0.25,
          0.0,
          10.0,
          50.0,
          0.25,
          1.0,
          20.0,
          50.0,
          0.25,
          0.0,
          30.0,
          50.0,
          0.25,
          1.0,
          40.0,
          50.0,
        ]));
    const bias = builder.constant(
        {type: 'float32', dimensions: [4]},
        new Float32Array([6000, 7000, 8000, 9000]));
    const expected = [6010, 7046, 11000, 9000];
    const conv = builder.conv2d(input, filter, {layout: 'nhwc', groups: 4});
    const output = builder.add(conv, bias);
    const model = builder.createModel({output});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input': {buffer: inputBuffer}});
    utils.checkShape(outputs.output.dimensions, [1, 1, 1, 4]);
    utils.checkValue(outputs.output.buffer, expected);
  });

  it('conv2d depthwise nchw', async function() {
    // It is based on Android NNAPI CTS: V1_2/depthwise_conv2d_v1_2.mod.py
    const builder = nn.createModelBuilder();
    const input =
        builder.input('input', {type: 'float32', dimensions: [1, 4, 2, 2]});
    const inputBuffer = new Float32Array(
        [10, 10, 10, 10, 21, 22, 23, 24, 10, 20, 30, 40, 0, 0, 0, 0]);
    const filter = builder.constant(
        {type: 'float32', dimensions: [4, 1, 2, 2]}, new Float32Array([
          0.25,
          0.25,
          0.25,
          0.25,
          0.0,
          1.0,
          0.0,
          1.0,
          10.0,
          20.0,
          30.0,
          40.0,
          50.0,
          50.0,
          50.0,
          50.0,
        ]));
    const bias = builder.constant(
        {type: 'float32', dimensions: [1, 4, 1, 1]},
        new Float32Array([6000, 7000, 8000, 9000]));
    const expected = [6010, 7046, 11000, 9000];
    const conv = builder.conv2d(input, filter, {groups: 4});
    const output = builder.add(conv, bias);
    const model = builder.createModel({output});
    const compilation = await model.compile();
    const outputs = await compilation.compute({'input': {buffer: inputBuffer}});
    utils.checkShape(outputs.output.dimensions, [1, 4, 1, 1]);
    utils.checkValue(outputs.output.buffer, expected);
  });
});
