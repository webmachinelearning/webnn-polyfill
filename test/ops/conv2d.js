'use strict';
import * as utils from '../utils.js';

describe('test conv2d', function() {
  const context = navigator.ml.createContext();

  async function testConv2d(
      input, filter, expected, options = {}, bias = undefined,
      activation = undefined) {
    const builder = new MLGraphBuilder(context);
    const x = builder.input('x', {type: 'float32', dimensions: input.shape});
    const w = builder.constant({type: 'float32', dimensions: filter.shape},
        filter.data);
    let y = builder.conv2d(x, w, options);
    if (bias !== undefined) {
      const b = builder.constant({type: 'float32', dimensions: bias.shape},
          bias.data);
      y = builder.add(y, b);
    }
    if (activation !== undefined) {
      if (activation === 'RELU') {
        y = builder.relu(y);
      }
    }
    const graph = await builder.build({y});
    const inputs = {'x': {data: input.data}};
    const outputs = await graph.compute(inputs);
    utils.checkShape(outputs.y.dimensions, expected.shape);
    utils.checkValue(outputs.y.data, expected.data);
  }

  it('conv2d with padding default', async function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Float32Array(9).fill(1),
    };
    const options = {padding: [1, 1, 1, 1]};
    const expected = {
      shape: [1, 1, 5, 5],
      data: [
        12,  21, 27, 33,  24,  33,  54,  63, 72,  51,  63,  99, 108,
        117, 81, 93, 144, 153, 162, 111, 72, 111, 117, 123, 84,
      ],
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with padding nchw oihw', async function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Float32Array(9).fill(1),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nchw',
      filterLayout: 'oihw',
    };
    const expected = {
      shape: [1, 1, 5, 5],
      data: [
        12,  21, 27, 33,  24,  33,  54,  63, 72,  51,  63,  99, 108,
        117, 81, 93, 144, 153, 162, 111, 72, 111, 117, 123, 84,
      ],
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with padding nchw hwio', async function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Float32Array(9).fill(1),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nchw',
      filterLayout: 'hwio',
    };
    const expected = {
      shape: [1, 1, 5, 5],
      data: [
        12,  21, 27, 33,  24,  33,  54,  63, 72,  51,  63,  99, 108,
        117, 81, 93, 144, 153, 162, 111, 72, 111, 117, 123, 84,
      ],
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with padding nchw ohwi', async function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    const expected = {
      shape: [1, 1, 5, 5],
      data: [
        12,  21, 27, 33,  24,  33,  54,  63, 72,  51,  63,  99, 108,
        117, 81, 93, 144, 153, 162, 111, 72, 111, 117, 123, 84,
      ],
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with padding nchw ihwo', async function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nchw',
      filterLayout: 'ihwo',
    };
    const expected = {
      shape: [1, 1, 5, 5],
      data: [
        12,  21, 27, 33,  24,  33,  54,  63, 72,  51,  63,  99, 108,
        117, 81, 93, 144, 153, 162, 111, 72, 111, 117, 123, 84,
      ],
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with padding nhwc oihw', async function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Float32Array(9).fill(1),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nhwc',
      filterLayout: 'oihw',
    };
    const expected = {
      shape: [1, 5, 5, 1],
      data: [
        12,  21, 27, 33,  24,  33,  54,  63, 72,  51,  63,  99, 108,
        117, 81, 93, 144, 153, 162, 111, 72, 111, 117, 123, 84,
      ],
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with padding nhwc hwio', async function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Float32Array(9).fill(1),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nhwc',
      filterLayout: 'hwio',
    };
    const expected = {
      shape: [1, 5, 5, 1],
      data: [
        12,  21, 27, 33,  24,  33,  54,  63, 72,  51,  63,  99, 108,
        117, 81, 93, 144, 153, 162, 111, 72, 111, 117, 123, 84,
      ],
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with padding nhwc ohwi', async function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    const expected = {
      shape: [1, 5, 5, 1],
      data: [
        12,  21, 27, 33,  24,  33,  54,  63, 72,  51,  63,  99, 108,
        117, 81, 93, 144, 153, 162, 111, 72, 111, 117, 123, 84,
      ],
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with padding nhwc ihwo', async function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nhwc',
      filterLayout: 'ihwo',
    };
    const expected = {
      shape: [1, 5, 5, 1],
      data: [
        12,  21, 27, 33,  24,  33,  54,  63, 72,  51,  63,  99, 108,
        117, 81, 93, 144, 153, 162, 111, 72, 111, 117, 123, 84,
      ],
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d without padding default', async function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [54., 63., 72., 99., 108., 117., 144., 153., 162.],
    };
    await testConv2d(input, filter, expected);
  });

  it('conv2d without padding nchw hwio', async function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [54., 63., 72., 99., 108., 117., 144., 153., 162.],
    };
    const options = {
      inputLayout: 'nchw',
      filterLayout: 'hwio',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d without padding nchw ohwi', async function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [54., 63., 72., 99., 108., 117., 144., 153., 162.],
    };
    const options = {
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d without padding nchw ihwo', async function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [54., 63., 72., 99., 108., 117., 144., 153., 162.],
    };
    const options = {
      inputLayout: 'nchw',
      filterLayout: 'ihwo',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d without padding nhwc oihw', async function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Float32Array(9).fill(1),
    };
    const options = {
      inputLayout: 'nhwc',
      filterLayout: 'oihw',
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [54., 63., 72., 99., 108., 117., 144., 153., 162.],
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d without padding nhwc hwio', async function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Float32Array(9).fill(1),
    };
    const options = {
      inputLayout: 'nhwc',
      filterLayout: 'hwio',
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [54., 63., 72., 99., 108., 117., 144., 153., 162.],
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d without padding nhwc ohwi', async function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const options = {
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [54., 63., 72., 99., 108., 117., 144., 153., 162.],
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d without padding nhwc ihwo', async function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const options = {
      inputLayout: 'nhwc',
      filterLayout: 'ihwo',
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [54., 63., 72., 99., 108., 117., 144., 153., 162.],
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and padding default', async function() {
    const input = {
      shape: [1, 1, 7, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      ]),
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 4, 3],
      data: [12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.],
    };
    const options = {
      padding: [1, 1, 1, 1],
      strides: [2, 2],
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and padding nchw hwio', async function() {
    const input = {
      shape: [1, 1, 7, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      ]),
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 4, 3],
      data: [12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.],
    };
    const options = {
      padding: [1, 1, 1, 1],
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'hwio',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and padding nchw ohwi', async function() {
    const input = {
      shape: [1, 1, 7, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 4, 3],
      data: [12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.],
    };
    const options = {
      padding: [1, 1, 1, 1],
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and padding nchw ihwo', async function() {
    const input = {
      shape: [1, 1, 7, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 4, 3],
      data: [12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.],
    };
    const options = {
      padding: [1, 1, 1, 1],
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'ihwo',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and padding nhwc oihw', async function() {
    const input = {
      shape: [1, 7, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      ]),
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 4, 3, 1],
      data: [12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.],
    };
    const options = {
      padding: [1, 1, 1, 1],
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'oihw',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and padding nhwc hwio', async function() {
    const input = {
      shape: [1, 7, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      ]),
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 4, 3, 1],
      data: [12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.],
    };
    const options = {
      padding: [1, 1, 1, 1],
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'hwio',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and padding nhwc ohwi', async function() {
    const input = {
      shape: [1, 7, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 4, 3, 1],
      data: [12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.],
    };
    const options = {
      padding: [1, 1, 1, 1],
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and padding nhwc ihwo', async function() {
    const input = {
      shape: [1, 7, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 4, 3, 1],
      data: [12., 27., 24., 63., 108., 81., 123., 198., 141., 112., 177., 124.],
    };
    const options = {
      padding: [1, 1, 1, 1],
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'ihwo',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and asymetric padding default', async function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 1, 4, 2],
      data: new Float32Array(8).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [33, 45, 27, 104, 120, 66, 72, 80, 43],
    };
    const options = {
      padding: [1, 2, 0, 1],
      strides: [2, 2],
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and asymetric padding nchw hwio', async function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [4, 2, 1, 1],
      data: new Float32Array(8).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [33, 45, 27, 104, 120, 66, 72, 80, 43],
    };
    const options = {
      padding: [1, 2, 0, 1],
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'hwio',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and asymetric padding nchw ohwi', async function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 4, 2, 1],
      data: new Float32Array(8).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [33, 45, 27, 104, 120, 66, 72, 80, 43],
    };
    const options = {
      padding: [1, 2, 0, 1],
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and asymetric padding nchw ihwo', async function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 4, 2, 1],
      data: new Float32Array(8).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [33, 45, 27, 104, 120, 66, 72, 80, 43],
    };
    const options = {
      padding: [1, 2, 0, 1],
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'ihwo',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and asymetric padding nhwc oihw', async function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 1, 4, 2],
      data: new Float32Array(8).fill(1),
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [33, 45, 27, 104, 120, 66, 72, 80, 43],
    };
    const options = {
      padding: [1, 2, 0, 1],
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'oihw',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and asymetric padding nhwc hwio', async function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [4, 2, 1, 1],
      data: new Float32Array(8).fill(1),
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [33, 45, 27, 104, 120, 66, 72, 80, 43],
    };
    const options = {
      padding: [1, 2, 0, 1],
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'hwio',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and asymetric padding nhwc ohwi', async function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 4, 2, 1],
      data: new Float32Array(8).fill(1),
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [33, 45, 27, 104, 120, 66, 72, 80, 43],
    };
    const options = {
      padding: [1, 2, 0, 1],
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with strides=2 and asymetric padding nhwc ihwo', async function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 4, 2, 1],
      data: new Float32Array(8).fill(1),
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [33, 45, 27, 104, 120, 66, 72, 80, 43],
    };
    const options = {
      padding: [1, 2, 0, 1],
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'ihwo',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-lower default', async function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [12., 27., 24., 63., 108., 81., 72., 117., 84.],
    };
    const options = {
      autoPad: 'same-lower',
      strides: [2, 2],
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-lower nchw hwio', async function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [12., 27., 24., 63., 108., 81., 72., 117., 84.],
    };
    const options = {
      autoPad: 'same-lower',
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'hwio',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-lower nchw ohwi', async function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [12., 27., 24., 63., 108., 81., 72., 117., 84.],
    };
    const options = {
      autoPad: 'same-lower',
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-lower nchw ihwo', async function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [12., 27., 24., 63., 108., 81., 72., 117., 84.],
    };
    const options = {
      autoPad: 'same-lower',
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'ihwo',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-lower nhwc oihw', async function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [12., 27., 24., 63., 108., 81., 72., 117., 84.],
    };
    const options = {
      autoPad: 'same-lower',
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'oihw',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-lower nhwc hwio', async function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [12., 27., 24., 63., 108., 81., 72., 117., 84.],
    };
    const options = {
      autoPad: 'same-lower',
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'hwio',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-lower nhwc ohwi', async function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [12., 27., 24., 63., 108., 81., 72., 117., 84.],
    };
    const options = {
      autoPad: 'same-lower',
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-lower nhwc ihwo', async function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 3, 3, 1],
      data: [12., 27., 24., 63., 108., 81., 72., 117., 84.],
    };
    const options = {
      autoPad: 'same-lower',
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'ihwo',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-upper default', async function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 3, 3],
      data: [12., 27., 24., 63., 108., 81., 72., 117., 84.],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-upper nchw hwio', async function() {
    const input = {
      shape: [1, 1, 4, 4],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
      ]),
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 2, 2],
      data: [45., 39., 66., 50.],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'hwio',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-upper nchw ohwi', async function() {
    const input = {
      shape: [1, 1, 4, 4],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 2, 2],
      data: [45., 39., 66., 50.],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-upper nchw ihwo', async function() {
    const input = {
      shape: [1, 1, 4, 4],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 1, 2, 2],
      data: [45., 39., 66., 50.],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'ihwo',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-upper nhwc oihw', async function() {
    const input = {
      shape: [1, 4, 4, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
      ]),
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 2, 2, 1],
      data: [45., 39., 66., 50.],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'oihw',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-upper nhwc hwio', async function() {
    const input = {
      shape: [1, 4, 4, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
      ]),
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 2, 2, 1],
      data: [45., 39., 66., 50.],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'hwio',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-upper nhwc ohwi', async function() {
    const input = {
      shape: [1, 4, 4, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 2, 2, 1],
      data: [45., 39., 66., 50.],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d with autopad same-upper nhwc ihwo', async function() {
    const input = {
      shape: [1, 4, 4, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const expected = {
      shape: [1, 2, 2, 1],
      data: [45., 39., 66., 50.],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'ihwo',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('fused depthwise conv2d default', async function() {
    // It is based on Android NNAPI CTS: V1_2/depthwise_conv2d_v1_2.mod.py
    const input = {
      shape: [1, 4, 2, 2],
      data: new Float32Array([
        10, 10, 10, 10, 21, 22, 23, 24, 10, 20, 30, 40, 0, 0, 0, 0,
      ]),
    };
    const filter = {
      shape: [4, 1, 2, 2],
      data: new Float32Array([
        0.25, 0.25, 0.25, 0.25, 0.0,  1.0,  0.0,  1.0,
        10.0, 20.0, 30.0, 40.0, 50.0, 50.0, 50.0, 50.0,
      ]),
    };
    const bias = {
      shape: [1, 4, 1, 1],
      data: new Float32Array([6000, 7000, 8000, 9000]),
    };
    const expected = {
      shape: [1, 4, 1, 1],
      data: [6010, 7046, 11000, 9000],
    };
    const options = {groups: 4};
    await testConv2d(input, filter, expected, options, bias);
  });

  it('fused depthwise conv2d nchw hwio', async function() {
    // It is based on Android NNAPI CTS: V1_2/depthwise_conv2d_v1_2.mod.py
    const input = {
      shape: [1, 4, 2, 2],
      data: new Float32Array([
        10, 10, 10, 10, 21, 22, 23, 24, 10, 20, 30, 40, 0, 0, 0, 0,
      ]),
    };
    const filter = {
      shape: [2, 2, 1, 4],
      data: new Float32Array([
        0.25, 0.0, 10.0, 50.0, 0.25, 1.0, 20.0, 50.0,
        0.25, 0.0, 30.0, 50.0, 0.25, 1.0, 40.0, 50.0,
      ]),
    };
    const bias = {
      shape: [1, 4, 1, 1],
      data: new Float32Array([6000, 7000, 8000, 9000]),
    };
    const expected = {
      shape: [1, 4, 1, 1],
      data: [6010, 7046, 11000, 9000],
    };
    const options = {
      groups: 4,
      inputLayout: 'nchw',
      filterLayout: 'hwio',
    };
    await testConv2d(input, filter, expected, options, bias);
  });

  it('fused depthwise conv2d nchw ohwi', async function() {
    // It is based on Android NNAPI CTS: V1_2/depthwise_conv2d_v1_2.mod.py
    const input = {
      shape: [1, 4, 2, 2],
      data: new Float32Array([
        10, 10, 10, 10, 21, 22, 23, 24, 10, 20, 30, 40, 0, 0, 0, 0,
      ]),
    };
    const filter = {
      shape: [4, 2, 2, 1],
      data: new Float32Array([
        0.25, 0.25, 0.25, 0.25, 0.0,  1.0,  0.0,  1.0,
        10.0, 20.0, 30.0, 40.0, 50.0, 50.0, 50.0, 50.0,
      ]),
    };
    const bias = {
      shape: [1, 4, 1, 1],
      data: new Float32Array([6000, 7000, 8000, 9000]),
    };
    const expected = {
      shape: [1, 4, 1, 1],
      data: [6010, 7046, 11000, 9000],
    };
    const options = {
      groups: 4,
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    await testConv2d(input, filter, expected, options, bias);
  });

  it('fused depthwise conv2d nchw ihwo', async function() {
    // It is based on Android NNAPI CTS: V1_2/depthwise_conv2d_v1_2.mod.py
    const input = {
      shape: [1, 4, 2, 2],
      data: new Float32Array([
        10, 10, 10, 10, 21, 22, 23, 24, 10, 20, 30, 40, 0, 0, 0, 0,
      ]),
    };
    const filter = {
      shape: [1, 2, 2, 4],
      data: new Float32Array([
        0.25, 0.0, 10.0, 50.0, 0.25, 1.0, 20.0, 50.0,
        0.25, 0.0, 30.0, 50.0, 0.25, 1.0, 40.0, 50.0,
      ]),
    };
    const bias = {
      shape: [1, 4, 1, 1],
      data: new Float32Array([6000, 7000, 8000, 9000]),
    };
    const expected = {
      shape: [1, 4, 1, 1],
      data: [6010, 7046, 11000, 9000],
    };
    const options = {
      groups: 4,
      inputLayout: 'nchw',
      filterLayout: 'ihwo',
    };
    await testConv2d(input, filter, expected, options, bias);
  });

  it('fused depthwise conv2d nhwc oihw', async function() {
    // It is based on Android NNAPI CTS: V1_2/depthwise_conv2d_v1_2.mod.py
    const input = {
      shape: [1, 2, 2, 4],
      data: new Float32Array([
        10, 21, 10, 0, 10, 22, 20, 0, 10, 23, 30, 0, 10, 24, 40, 0,
      ]),
    };
    const filter = {
      shape: [4, 1, 2, 2],
      data: new Float32Array([
        0.25, 0.25, 0.25, 0.25, 0.0,  1.0,  0.0,  1.0,
        10.0, 20.0, 30.0, 40.0, 50.0, 50.0, 50.0, 50.0,
      ]),
    };
    const bias = {
      shape: [4],
      data: new Float32Array([6000, 7000, 8000, 9000]),
    };
    const expected = {
      shape: [1, 1, 1, 4],
      data: [6010, 7046, 11000, 9000],
    };
    const options = {
      groups: 4,
      inputLayout: 'nhwc',
      filterLayout: 'oihw',
    };
    await testConv2d(input, filter, expected, options, bias);
  });

  it('fused depthwise conv2d nhwc hwio', async function() {
    // It is based on Android NNAPI CTS: V1_2/depthwise_conv2d_v1_2.mod.py
    const input = {
      shape: [1, 2, 2, 4],
      data: new Float32Array([
        10, 21, 10, 0, 10, 22, 20, 0, 10, 23, 30, 0, 10, 24, 40, 0,
      ]),
    };
    const filter = {
      shape: [2, 2, 1, 4],
      data: new Float32Array([
        0.25, 0.0, 10.0, 50.0, 0.25, 1.0, 20.0, 50.0,
        0.25, 0.0, 30.0, 50.0, 0.25, 1.0, 40.0, 50.0,
      ]),
    };
    const bias = {
      shape: [4],
      data: new Float32Array([6000, 7000, 8000, 9000]),
    };
    const expected = {
      shape: [1, 1, 1, 4],
      data: [6010, 7046, 11000, 9000],
    };
    const options = {
      groups: 4,
      inputLayout: 'nhwc',
      filterLayout: 'hwio',
    };
    await testConv2d(input, filter, expected, options, bias);
  });

  it('fused depthwise conv2d nhwc ohwi', async function() {
    // It is based on Android NNAPI CTS: V1_2/depthwise_conv2d_v1_2.mod.py
    const input = {
      shape: [1, 2, 2, 4],
      data: new Float32Array([
        10, 21, 10, 0, 10, 22, 20, 0, 10, 23, 30, 0, 10, 24, 40, 0,
      ]),
    };
    const filter = {
      shape: [4, 2, 2, 1],
      data: new Float32Array([
        0.25, 0.25, 0.25, 0.25, 0.0,  1.0,  0.0,  1.0,
        10.0, 20.0, 30.0, 40.0, 50.0, 50.0, 50.0, 50.0,
      ]),
    };
    const bias = {
      shape: [4],
      data: new Float32Array([6000, 7000, 8000, 9000]),
    };
    const expected = {
      shape: [1, 1, 1, 4],
      data: [6010, 7046, 11000, 9000],
    };
    const options = {
      groups: 4,
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    await testConv2d(input, filter, expected, options, bias);
  });

  it('fused depthwise conv2d nhwc ihwo', async function() {
    // It is based on Android NNAPI CTS: V1_2/depthwise_conv2d_v1_2.mod.py
    const input = {
      shape: [1, 2, 2, 4],
      data: new Float32Array([
        10, 21, 10, 0, 10, 22, 20, 0, 10, 23, 30, 0, 10, 24, 40, 0,
      ]),
    };
    const filter = {
      shape: [1, 2, 2, 4],
      data: new Float32Array([
        0.25, 0.0, 10.0, 50.0, 0.25, 1.0, 20.0, 50.0,
        0.25, 0.0, 30.0, 50.0, 0.25, 1.0, 40.0, 50.0,
      ]),
    };
    const bias = {
      shape: [4],
      data: new Float32Array([6000, 7000, 8000, 9000]),
    };
    const expected = {
      shape: [1, 1, 1, 4],
      data: [6010, 7046, 11000, 9000],
    };
    const options = {
      groups: 4,
      inputLayout: 'nhwc',
      filterLayout: 'ihwo',
    };
    await testConv2d(input, filter, expected, options, bias);
  });

  it('depthwise conv2d nchw oihw', async function() {
    const input = {
      shape: [1, 4, 2, 2],
      data: new Float32Array([
        10, 10, 10, 10, 21, 22, 23, 24, 10, 20, 30, 40, 0, 0, 0, 0,
      ]),
    };
    const filter = {
      shape: [4, 1, 2, 2],
      data: new Float32Array([
        0.25, 0.25, 0.25, 0.25, 0.0,  1.0,  0.0,  1.0,
        10.0, 20.0, 30.0, 40.0, 50.0, 50.0, 50.0, 50.0,
      ]),
    };
    const expected = {
      shape: [1, 4, 1, 1],
      data: [10, 46, 3000, 0],
    };
    const options = {
      groups: 4,
      inputLayout: 'nchw',
      filterLayout: 'oihw',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('fused conv2d with padding default', async function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Float32Array(9).fill(1),
    };
    const bias = {
      shape: [1],
      data: new Float32Array([-100]),
    };
    const options = {
      padding: [1, 1, 1, 1],
    };
    const expected = {
      shape: [1, 1, 5, 5],
      data: [
        0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
        17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.,
      ],
    };
    await testConv2d(input, filter, expected, options, bias, 'RELU');
  });

  it('fused conv2d with padding nchw hwio', async function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Float32Array(9).fill(1),
    };
    const bias = {
      shape: [1],
      data: new Float32Array([-100]),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nchw',
      filterLayout: 'hwio',
    };
    const expected = {
      shape: [1, 1, 5, 5],
      data: [
        0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
        17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.,
      ],
    };
    await testConv2d(input, filter, expected, options, bias, 'RELU');
  });

  it('fused conv2d with padding nchw ohwi', async function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const bias = {
      shape: [1],
      data: new Float32Array([-100]),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    const expected = {
      shape: [1, 1, 5, 5],
      data: [
        0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
        17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.,
      ],
    };
    await testConv2d(input, filter, expected, options, bias, 'RELU');
  });

  it('fused conv2d with padding nchw ihwo', async function() {
    const input = {
      shape: [1, 1, 5, 5],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const bias = {
      shape: [1],
      data: new Float32Array([-100]),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nchw',
      filterLayout: 'ihwo',
    };
    const expected = {
      shape: [1, 1, 5, 5],
      data: [
        0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
        17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.,
      ],
    };
    await testConv2d(input, filter, expected, options, bias, 'RELU');
  });

  it('fused conv2d with padding nhwc oihw', async function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 1, 3, 3],
      data: new Float32Array(9).fill(1),
    };
    const bias = {
      shape: [1],
      data: new Float32Array([-100]),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nhwc',
      filterLayout: 'oihw',
    };
    const expected = {
      shape: [1, 5, 5, 1],
      data: [
        0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
        17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.,
      ],
    };
    await testConv2d(input, filter, expected, options, bias, 'RELU');
  });

  it('fused conv2d with padding nhwc hwio', async function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [3, 3, 1, 1],
      data: new Float32Array(9).fill(1),
    };
    const bias = {
      shape: [1],
      data: new Float32Array([-100]),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nhwc',
      filterLayout: 'hwio',
    };
    const expected = {
      shape: [1, 5, 5, 1],
      data: [
        0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
        17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.,
      ],
    };
    await testConv2d(input, filter, expected, options, bias, 'RELU');
  });

  it('fused conv2d with padding nhwc ohwi', async function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const bias = {
      shape: [1],
      data: new Float32Array([-100]),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    const expected = {
      shape: [1, 5, 5, 1],
      data: [
        0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
        17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.,
      ],
    };
    await testConv2d(input, filter, expected, options, bias, 'RELU');
  });

  it('fused conv2d with padding nhwc ihwo', async function() {
    const input = {
      shape: [1, 5, 5, 1],
      data: new Float32Array([
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
      ]),
    };
    const filter = {
      shape: [1, 3, 3, 1],
      data: new Float32Array(9).fill(1),
    };
    const bias = {
      shape: [1],
      data: new Float32Array([-100]),
    };
    const options = {
      padding: [1, 1, 1, 1],
      inputLayout: 'nhwc',
      filterLayout: 'ihwo',
    };
    const expected = {
      shape: [1, 5, 5, 1],
      data: [
        0.,  0., 0., 0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0., 8.,
        17., 0., 0., 44., 53., 62., 11., 0., 11., 17., 23., 0.,
      ],
    };
    await testConv2d(input, filter, expected, options, bias, 'RELU');
  });

  it('conv2d transpose default', async function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [1, 2, 3, 3],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 2, 5, 5],
      data: [
        0.,  1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36.,
        27., 15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.,  0.,
        1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36., 27.,
        15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.,
      ],
    };
    const options = {transpose: true};
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose nchw hwio', async function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [3, 3, 2, 1],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 2, 5, 5],
      data: [
        0.,  1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36.,
        27., 15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.,  0.,
        1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36., 27.,
        15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.,
      ],
    };
    const options = {
      transpose: true,
      inputLayout: 'nchw',
      filterLayout: 'hwio',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose nchw ohwi', async function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [1, 3, 3, 2],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 2, 5, 5],
      data: [
        0.,  1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36.,
        27., 15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.,  0.,
        1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36., 27.,
        15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.,
      ],
    };
    const options = {
      transpose: true,
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose nchw ihwo', async function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [2, 3, 3, 1],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 2, 5, 5],
      data: [
        0.,  1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36.,
        27., 15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.,  0.,
        1.,  3.,  3.,  2.,  3.,  8.,  15., 12., 7.,  9.,  21., 36., 27.,
        15., 9.,  20., 33., 24., 13., 6.,  13., 21., 15., 8.,
      ],
    };
    const options = {
      transpose: true,
      inputLayout: 'nchw',
      filterLayout: 'ihwo',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose nhwc oihw', async function() {
    const input = {
      shape: [1, 3, 3, 1],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [1, 2, 3, 3],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 5, 5, 2],
      data: [
        0.,  0.,  1.,  1.,  3.,  3.,  3.,  3.,  2.,  2.,  3.,  3.,  8.,
        8.,  15., 15., 12., 12., 7.,  7.,  9.,  9.,  21., 21., 36., 36.,
        27., 27., 15., 15., 9.,  9.,  20., 20., 33., 33., 24., 24., 13.,
        13., 6.,  6.,  13., 13., 21., 21., 15., 15., 8.,  8.,
      ],
    };
    const options = {
      transpose: true,
      inputLayout: 'nhwc',
      filterLayout: 'oihw',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose nhwc hwio', async function() {
    const input = {
      shape: [1, 3, 3, 1],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [3, 3, 2, 1],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 5, 5, 2],
      data: [
        0.,  0.,  1.,  1.,  3.,  3.,  3.,  3.,  2.,  2.,  3.,  3.,  8.,
        8.,  15., 15., 12., 12., 7.,  7.,  9.,  9.,  21., 21., 36., 36.,
        27., 27., 15., 15., 9.,  9.,  20., 20., 33., 33., 24., 24., 13.,
        13., 6.,  6.,  13., 13., 21., 21., 15., 15., 8.,  8.,
      ],
    };
    const options = {
      transpose: true,
      inputLayout: 'nhwc',
      filterLayout: 'hwio',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose nhwc ohwi', async function() {
    const input = {
      shape: [1, 3, 3, 1],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [1, 3, 3, 2],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 5, 5, 2],
      data: [
        0.,  0.,  1.,  1.,  3.,  3.,  3.,  3.,  2.,  2.,  3.,  3.,  8.,
        8.,  15., 15., 12., 12., 7.,  7.,  9.,  9.,  21., 21., 36., 36.,
        27., 27., 15., 15., 9.,  9.,  20., 20., 33., 33., 24., 24., 13.,
        13., 6.,  6.,  13., 13., 21., 21., 15., 15., 8.,  8.,
      ],
    };
    const options = {
      transpose: true,
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose nhwc ihwo', async function() {
    const input = {
      shape: [1, 3, 3, 1],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [2, 3, 3, 1],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 5, 5, 2],
      data: [
        0.,  0.,  1.,  1.,  3.,  3.,  3.,  3.,  2.,  2.,  3.,  3.,  8.,
        8.,  15., 15., 12., 12., 7.,  7.,  9.,  9.,  21., 21., 36., 36.,
        27., 27., 15., 15., 9.,  9.,  20., 20., 33., 33., 24., 24., 13.,
        13., 6.,  6.,  13., 13., 21., 21., 15., 15., 8.,  8.,
      ],
    };
    const options = {
      transpose: true,
      inputLayout: 'nhwc',
      filterLayout: 'ihwo',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose output shape default', async function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [1, 2, 3, 3],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 2, 10, 8],
      data: [
        0., 0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 0., 0., 0.,  0., 0.,  0., 0., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 0., 0., 0.,  0., 0.,  0., 0., 0.,
      ],
    };
    const options = {
      strides: [3, 2],
      outputSizes: [10, 8],
      transpose: true,
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose output shape nchw hwio', async function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [3, 3, 2, 1],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 2, 10, 8],
      data: [
        0., 0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 0., 0., 0.,  0., 0.,  0., 0., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 0., 0., 0.,  0., 0.,  0., 0., 0.,
      ],
    };
    const options = {
      strides: [3, 2],
      outputSizes: [10, 8],
      transpose: true,
      inputLayout: 'nchw',
      filterLayout: 'hwio',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose output shape nchw ohwi', async function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [1, 3, 3, 2],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 2, 10, 8],
      data: [
        0., 0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 0., 0., 0.,  0., 0.,  0., 0., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 0., 0., 0.,  0., 0.,  0., 0., 0.,
      ],
    };
    const options = {
      strides: [3, 2],
      outputSizes: [10, 8],
      transpose: true,
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose output shape nchw ihwo', async function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [2, 3, 3, 1],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 2, 10, 8],
      data: [
        0., 0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 0., 0., 0.,  0., 0.,  0., 0., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 0., 0., 0.,  0., 0.,  0., 0., 0.,
      ],
    };
    const options = {
      strides: [3, 2],
      outputSizes: [10, 8],
      transpose: true,
      inputLayout: 'nchw',
      filterLayout: 'ihwo',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose output shape nhwc oihw', async function() {
    const input = {
      shape: [1, 3, 3, 1],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [1, 2, 3, 3],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 10, 8, 2],
      data: [
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      ],
    };
    const options = {
      strides: [3, 2],
      outputSizes: [10, 8],
      transpose: true,
      inputLayout: 'nhwc',
      filterLayout: 'oihw',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose output shape nhwc hwio', async function() {
    const input = {
      shape: [1, 3, 3, 1],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [3, 3, 2, 1],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 10, 8, 2],
      data: [
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      ],
    };
    const options = {
      strides: [3, 2],
      outputSizes: [10, 8],
      transpose: true,
      inputLayout: 'nhwc',
      filterLayout: 'hwio',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose output shape nhwc ohwi', async function() {
    const input = {
      shape: [1, 3, 3, 1],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [1, 3, 3, 2],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 10, 8, 2],
      data: [
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      ],
    };
    const options = {
      strides: [3, 2],
      outputSizes: [10, 8],
      transpose: true,
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose output shape nhwc ihwo', async function() {
    const input = {
      shape: [1, 3, 3, 1],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [2, 3, 3, 1],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 10, 8, 2],
      data: [
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      ],
    };
    const options = {
      strides: [3, 2],
      outputSizes: [10, 8],
      transpose: true,
      inputLayout: 'nhwc',
      filterLayout: 'ihwo',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose out pad default', async function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [1, 2, 3, 3],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 2, 10, 8],
      data: [
        0., 0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 0., 0., 0.,  0., 0.,  0., 0., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 0., 0., 0.,  0., 0.,  0., 0., 0.,
      ],
    };
    const options = {
      strides: [3, 2],
      outputPadding: [1, 1],
      transpose: true,
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose out pad nchw hwio', async function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [3, 3, 2, 1],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 2, 10, 8],
      data: [
        0., 0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 0., 0., 0.,  0., 0.,  0., 0., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 0., 0., 0.,  0., 0.,  0., 0., 0.,
      ],
    };
    const options = {
      strides: [3, 2],
      outputPadding: [1, 1],
      transpose: true,
      inputLayout: 'nchw',
      filterLayout: 'hwio',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose out pad nchw ohwi', async function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [1, 3, 3, 2],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 2, 10, 8],
      data: [
        0., 0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 0., 0., 0.,  0., 0.,  0., 0., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 0., 0., 0.,  0., 0.,  0., 0., 0.,
      ],
    };
    const options = {
      strides: [3, 2],
      outputPadding: [1, 1],
      transpose: true,
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose out pad nchw ihwo', async function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [2, 3, 3, 1],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 2, 10, 8],
      data: [
        0., 0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 0., 0., 0.,  0., 0.,  0., 0., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 0., 0., 1.,  1., 3.,  2., 2., 0.,
        0., 0., 1.,  1., 3.,  2., 2., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        3., 3., 7.,  4., 9.,  5., 5., 0., 3., 3., 7.,  4., 9.,  5., 5., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 6., 6., 13., 7., 15., 8., 8., 0.,
        6., 6., 13., 7., 15., 8., 8., 0., 0., 0., 0.,  0., 0.,  0., 0., 0.,
      ],
    };
    const options = {
      strides: [3, 2],
      outputPadding: [1, 1],
      transpose: true,
      inputLayout: 'nchw',
      filterLayout: 'ihwo',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose out pad nhwc oihw', async function() {
    const input = {
      shape: [1, 3, 3, 1],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [1, 2, 3, 3],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 10, 8, 2],
      data: [
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      ],
    };
    const options = {
      strides: [3, 2],
      outputPadding: [1, 1],
      transpose: true,
      inputLayout: 'nhwc',
      filterLayout: 'oihw',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose out pad nhwc hwio', async function() {
    const input = {
      shape: [1, 3, 3, 1],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [3, 3, 2, 1],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 10, 8, 2],
      data: [
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      ],
    };
    const options = {
      strides: [3, 2],
      outputPadding: [1, 1],
      transpose: true,
      inputLayout: 'nhwc',
      filterLayout: 'hwio',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose out pad nhwc ohwi', async function() {
    const input = {
      shape: [1, 3, 3, 1],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [1, 3, 3, 2],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 10, 8, 2],
      data: [
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      ],
    };
    const options = {
      strides: [3, 2],
      outputPadding: [1, 1],
      transpose: true,
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose out pad nhwc ihwo', async function() {
    const input = {
      shape: [1, 3, 3, 1],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [2, 3, 3, 1],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 10, 8, 2],
      data: [
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 5., 5., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      ],
    };
    const options = {
      strides: [3, 2],
      outputPadding: [1, 1],
      transpose: true,
      inputLayout: 'nhwc',
      filterLayout: 'ihwo',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose autopad same default', async function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [1, 2, 3, 3],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 2, 6, 6],
      data: [
        0., 0.,  1.,  1.,  3.,  2.,  0., 0.,  1.,  1., 3.,  2.,  3.,  3.,  8.,
        5., 12., 7.,  3.,  3.,  7.,  4., 9.,  5.,  9., 9.,  20., 11., 24., 13.,
        6., 6.,  13., 7.,  15., 8.,  0., 0.,  1.,  1., 3.,  2.,  0.,  0.,  1.,
        1., 3.,  2.,  3.,  3.,  8.,  5., 12., 7.,  3., 3.,  7.,  4.,  9.,  5.,
        9., 9.,  20., 11., 24., 13., 6., 6.,  13., 7., 15., 8.,
      ],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      transpose: true,
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose autopad same nchw hwio', async function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [3, 3, 2, 1],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 2, 6, 6],
      data: [
        0., 0.,  1.,  1.,  3.,  2.,  0., 0.,  1.,  1., 3.,  2.,  3.,  3.,  8.,
        5., 12., 7.,  3.,  3.,  7.,  4., 9.,  5.,  9., 9.,  20., 11., 24., 13.,
        6., 6.,  13., 7.,  15., 8.,  0., 0.,  1.,  1., 3.,  2.,  0.,  0.,  1.,
        1., 3.,  2.,  3.,  3.,  8.,  5., 12., 7.,  3., 3.,  7.,  4.,  9.,  5.,
        9., 9.,  20., 11., 24., 13., 6., 6.,  13., 7., 15., 8.,
      ],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      transpose: true,
      inputLayout: 'nchw',
      filterLayout: 'hwio',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose autopad same nchw ohwi', async function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [1, 3, 3, 2],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 2, 6, 6],
      data: [
        0., 0.,  1.,  1.,  3.,  2.,  0., 0.,  1.,  1., 3.,  2.,  3.,  3.,  8.,
        5., 12., 7.,  3.,  3.,  7.,  4., 9.,  5.,  9., 9.,  20., 11., 24., 13.,
        6., 6.,  13., 7.,  15., 8.,  0., 0.,  1.,  1., 3.,  2.,  0.,  0.,  1.,
        1., 3.,  2.,  3.,  3.,  8.,  5., 12., 7.,  3., 3.,  7.,  4.,  9.,  5.,
        9., 9.,  20., 11., 24., 13., 6., 6.,  13., 7., 15., 8.,
      ],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      transpose: true,
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose autopad same nchw ihwo', async function() {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [2, 3, 3, 1],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 2, 6, 6],
      data: [
        0., 0.,  1.,  1.,  3.,  2.,  0., 0.,  1.,  1., 3.,  2.,  3.,  3.,  8.,
        5., 12., 7.,  3.,  3.,  7.,  4., 9.,  5.,  9., 9.,  20., 11., 24., 13.,
        6., 6.,  13., 7.,  15., 8.,  0., 0.,  1.,  1., 3.,  2.,  0.,  0.,  1.,
        1., 3.,  2.,  3.,  3.,  8.,  5., 12., 7.,  3., 3.,  7.,  4.,  9.,  5.,
        9., 9.,  20., 11., 24., 13., 6., 6.,  13., 7., 15., 8.,
      ],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      transpose: true,
      inputLayout: 'nchw',
      filterLayout: 'ihwo',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose autopad same nhwc oihw', async function() {
    const input = {
      shape: [1, 3, 3, 1],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [1, 2, 3, 3],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 6, 6, 2],
      data: [
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 0., 0., 0.,
        0., 1., 1., 1., 1., 3., 3., 2., 2., 3., 3., 3., 3., 8., 8.,
        5., 5., 12, 12, 7., 7., 3., 3., 3., 3., 7., 7., 4., 4., 9.,
        9., 5., 5., 9., 9., 9., 9., 20, 20, 11, 11, 24, 24, 13, 13,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8.,
      ],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      transpose: true,
      inputLayout: 'nhwc',
      filterLayout: 'oihw',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose autopad same nhwc hwio', async function() {
    const input = {
      shape: [1, 3, 3, 1],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [3, 3, 2, 1],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 6, 6, 2],
      data: [
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 0., 0., 0.,
        0., 1., 1., 1., 1., 3., 3., 2., 2., 3., 3., 3., 3., 8., 8.,
        5., 5., 12, 12, 7., 7., 3., 3., 3., 3., 7., 7., 4., 4., 9.,
        9., 5., 5., 9., 9., 9., 9., 20, 20, 11, 11, 24, 24, 13, 13,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8.,
      ],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      transpose: true,
      inputLayout: 'nhwc',
      filterLayout: 'hwio',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose autopad same nhwc ohwi', async function() {
    const input = {
      shape: [1, 3, 3, 1],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [1, 3, 3, 2],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 6, 6, 2],
      data: [
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 0., 0., 0.,
        0., 1., 1., 1., 1., 3., 3., 2., 2., 3., 3., 3., 3., 8., 8.,
        5., 5., 12, 12, 7., 7., 3., 3., 3., 3., 7., 7., 4., 4., 9.,
        9., 5., 5., 9., 9., 9., 9., 20, 20, 11, 11, 24, 24, 13, 13,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8.,
      ],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      transpose: true,
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    await testConv2d(input, filter, expected, options);
  });

  it('conv2d transpose autopad same nhwc ihwo', async function() {
    const input = {
      shape: [1, 3, 3, 1],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [2, 3, 3, 1],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 6, 6, 2],
      data: [
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 0., 0., 0.,
        0., 1., 1., 1., 1., 3., 3., 2., 2., 3., 3., 3., 3., 8., 8.,
        5., 5., 12, 12, 7., 7., 3., 3., 3., 3., 7., 7., 4., 4., 9.,
        9., 5., 5., 9., 9., 9., 9., 20, 20, 11, 11, 24, 24, 13, 13,
        6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8.,
      ],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      transpose: true,
      inputLayout: 'nhwc',
      filterLayout: 'ihwo',
    };
    await testConv2d(input, filter, expected, options);
  });
});
