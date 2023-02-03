'use strict';
import * as utils from '../utils.js';

describe('test convTranspose2d', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  async function testConvTranspose2d(
      input, filter, expected, options = {}, bias = undefined,
      activation = undefined, fusion = false, activationOptions = {}) {
    const builder = new MLGraphBuilder(context);
    const x = builder.input('x', {type: 'float32', dimensions: input.shape});
    const w = builder.constant(
        {type: 'float32', dimensions: filter.shape}, filter.data);
    let b;
    if (bias !== undefined) {
      b = builder.constant(
          {type: 'float32', dimensions: bias.shape}, bias.data);
    }
    if (fusion) {
      if (b !== undefined) {
        options.bias = b;
      }
      if (activation !== undefined) {
        options.activation = utils.createActivation(
            builder, activation, undefined, activationOptions);
      }
    }
    let y = builder.convTranspose2d(x, w, options);
    if (!fusion) {
      if (b !== undefined) {
        if (options.inputLayout === undefined ||
            options.inputLayout === 'nchw') {
          b = builder.reshape(b, [1, null, 1, 1]);
        }
        y = builder.add(y, b);
      }
      if (activation !== undefined) {
        y = utils.createActivation(builder, activation, y, activationOptions);
      }
    }
    const graph = await builder.build({y});
    const inputs = {'x': input.data};
    const outputs = {'y': new Float32Array(utils.sizeOfShape(expected.shape))};
    await context.compute(graph, inputs, outputs);
    utils.checkValue(outputs.y, expected.data);
  }

  it('convTranspose2d default', async () => {
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
    await testConvTranspose2d(input, filter, expected);
  });

  it('convTranspose2d nchw hwoi', async () => {
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
      inputLayout: 'nchw',
      filterLayout: 'hwoi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d nchw ohwi', async () => {
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
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d nhwc iohw', async () => {
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
      inputLayout: 'nhwc',
      filterLayout: 'iohw',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d nhwc hwoi', async () => {
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
      inputLayout: 'nhwc',
      filterLayout: 'hwoi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d nhwc ohwi', async () => {
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
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d outputSizes default', async () => {
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
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d outputSizes nchw hwoi', async () => {
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
      inputLayout: 'nchw',
      filterLayout: 'hwoi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d outputSizes nchw ohwi', async () => {
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
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d outputSizes nhwc iohw', async () => {
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
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0.,
        0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0.,
        1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 3., 3., 3., 3., 7., 7.,
        4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4.,
        9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9.,
        5., 5., 5., 5., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8.,
        8., 8., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8.,
        0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      ],
    };
    const options = {
      strides: [3, 2],
      outputSizes: [10, 8],
      inputLayout: 'nhwc',
      filterLayout: 'iohw',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d outputSizes nhwc hwoi', async () => {
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
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0.,
        0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0.,
        1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 3., 3., 3., 3., 7., 7.,
        4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4.,
        9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9.,
        5., 5., 5., 5., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8.,
        8., 8., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8.,
        0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      ],
    };
    const options = {
      strides: [3, 2],
      outputSizes: [10, 8],
      inputLayout: 'nhwc',
      filterLayout: 'hwoi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d outputSizes nhwc ohwi', async () => {
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
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0.,
        0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0.,
        1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 3., 3., 3., 3., 7., 7.,
        4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4.,
        9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9.,
        5., 5., 5., 5., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8.,
        8., 8., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8.,
        0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      ],
    };
    const options = {
      strides: [3, 2],
      outputSizes: [10, 8],
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d out pad default', async () => {
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
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d out pad nchw hwoi', async () => {
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
      inputLayout: 'nchw',
      filterLayout: 'hwoi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d out pad nchw ohwi', async () => {
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
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d out pad nhwc iohw', async () => {
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
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0.,
        0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0.,
        1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 3., 3., 3., 3., 7., 7.,
        4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4.,
        9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9.,
        5., 5., 5., 5., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8.,
        8., 8., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8.,
        0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      ],
    };
    const options = {
      strides: [3, 2],
      outputPadding: [1, 1],
      inputLayout: 'nhwc',
      filterLayout: 'iohw',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d out pad nhwc hwoi', async () => {
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
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0.,
        0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0.,
        1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 3., 3., 3., 3., 7., 7.,
        4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4.,
        9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9.,
        5., 5., 5., 5., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8.,
        8., 8., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8.,
        0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      ],
    };
    const options = {
      strides: [3, 2],
      outputPadding: [1, 1],
      inputLayout: 'nhwc',
      filterLayout: 'hwoi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d out pad nhwc ohwi', async () => {
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
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0.,
        0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0.,
        1., 1., 1., 1., 3., 3., 2., 2., 2., 2., 0., 0., 3., 3., 3., 3., 7., 7.,
        4., 4., 9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4.,
        9., 9., 5., 5., 5., 5., 0., 0., 3., 3., 3., 3., 7., 7., 4., 4., 9., 9.,
        5., 5., 5., 5., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8.,
        8., 8., 0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8.,
        0., 0., 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8., 8., 8., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
      ],
    };
    const options = {
      strides: [3, 2],
      outputPadding: [1, 1],
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d autopad same-upper default', async () => {
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
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d autopad same-upper nchw hwoi', async () => {
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
      inputLayout: 'nchw',
      filterLayout: 'hwoi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d autopad same-upper nchw ohwi', async () => {
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
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d autopad same-upper nhwc iohw', async () => {
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
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 0., 0., 0., 0., 1., 1.,
        1., 1., 3., 3., 2., 2., 3., 3., 3., 3., 8., 8., 5., 5., 12, 12, 7., 7.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 9., 9., 9., 9., 20, 20,
        11, 11, 24, 24, 13, 13, 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8.,
      ],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'iohw',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d autopad same-upper nhwc hwoi', async () => {
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
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 0., 0., 0., 0., 1., 1.,
        1., 1., 3., 3., 2., 2., 3., 3., 3., 3., 8., 8., 5., 5., 12, 12, 7., 7.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 9., 9., 9., 9., 20, 20,
        11, 11, 24, 24, 13, 13, 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8.,
      ],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'hwoi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d autopad same-upper nhwc ohwi', async () => {
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
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 0., 0., 0., 0., 1., 1.,
        1., 1., 3., 3., 2., 2., 3., 3., 3., 3., 8., 8., 5., 5., 12, 12, 7., 7.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 9., 9., 9., 9., 20, 20,
        11, 11, 24, 24, 13, 13, 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8.,
      ],
    };
    const options = {
      autoPad: 'same-upper',
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d autopad explicit default padding', async () => {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [1, 2, 3, 3],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 7, 7, 2],
      data: [
        0, 0, 1, 1, 3, 2, 2,
        0, 0, 1, 1, 3, 2, 2,
        3, 3, 8, 5, 12, 7, 7,
        3, 3, 7, 4, 9, 5, 5,
        9, 9, 20, 11, 24, 13, 13,
        6, 6, 13, 7, 15, 8, 8,
        6, 6, 13, 7, 15, 8, 8,
        0, 0, 1, 1, 3, 2, 2,
        0, 0, 1, 1, 3, 2, 2,
        3, 3, 8, 5, 12, 7, 7,
        3, 3, 7, 4, 9, 5, 5,
        9, 9, 20, 11, 24, 13, 13,
        6, 6, 13, 7, 15, 8, 8,
        6, 6, 13, 7, 15, 8, 8,
      ],
    };
    const options = {
      autoPad: 'explicit',
      strides: [2, 2],
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d autopad explicit nchw iohw', async () => {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [1, 2, 3, 3],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 6, 6, 2],
      data: [
        0, 0, 1, 1, 3, 2,
        0, 0, 1, 1, 3, 2,
        3, 3, 8, 5, 12, 7,
        3, 3, 7, 4, 9, 5,
        9, 9, 20, 11, 24, 13,
        6, 6, 13, 7, 15, 8,
        0, 0, 1, 1, 3, 2,
        0, 0, 1, 1, 3, 2,
        3, 3, 8, 5, 12, 7,
        3, 3, 7, 4, 9, 5,
        9, 9, 20, 11, 24, 13,
        6, 6, 13, 7, 15, 8,
      ],
    };
    const options = {
      autoPad: 'explicit',
      padding: [0, 1, 0, 1],
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'iohw',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d autopad explicit nchw hwoi', async () => {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [3, 3, 2, 1],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 6, 6, 2],
      data: [
        0, 0, 1, 1, 3, 2,
        0, 0, 1, 1, 3, 2,
        3, 3, 8, 5, 12, 7,
        3, 3, 7, 4, 9, 5,
        9, 9, 20, 11, 24, 13,
        6, 6, 13, 7, 15, 8,
        0, 0, 1, 1, 3, 2,
        0, 0, 1, 1, 3, 2,
        3, 3, 8, 5, 12, 7,
        3, 3, 7, 4, 9, 5,
        9, 9, 20, 11, 24, 13,
        6, 6, 13, 7, 15, 8,
      ],
    };
    const options = {
      autoPad: 'explicit',
      padding: [0, 1, 0, 1],
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'hwoi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d autopad explicit nchw ohwi', async () => {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [2, 3, 3, 1],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 6, 6, 2],
      data: [
        0, 0, 1, 1, 3, 2,
        0, 0, 1, 1, 3, 2,
        3, 3, 8, 5, 12, 7,
        3, 3, 7, 4, 9, 5,
        9, 9, 20, 11, 24, 13,
        6, 6, 13, 7, 15, 8,
        0, 0, 1, 1, 3, 2,
        0, 0, 1, 1, 3, 2,
        3, 3, 8, 5, 12, 7,
        3, 3, 7, 4, 9, 5,
        9, 9, 20, 11, 24, 13,
        6, 6, 13, 7, 15, 8,
      ],
    };
    const options = {
      autoPad: 'explicit',
      padding: [0, 1, 0, 1],
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d autopad explicit nhwc iohw', async () => {
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
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 0., 0., 0., 0., 1., 1.,
        1., 1., 3., 3., 2., 2., 3., 3., 3., 3., 8., 8., 5., 5., 12, 12, 7., 7.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 9., 9., 9., 9., 20, 20,
        11, 11, 24, 24, 13, 13, 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8.,
      ],
    };
    const options = {
      autoPad: 'explicit',
      padding: [0, 1, 0, 1],
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'iohw',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d autopad explicit nhwc hwoi', async () => {
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
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 0., 0., 0., 0., 1., 1.,
        1., 1., 3., 3., 2., 2., 3., 3., 3., 3., 8., 8., 5., 5., 12, 12, 7., 7.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 9., 9., 9., 9., 20, 20,
        11, 11, 24, 24, 13, 13, 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8.,
      ],
    };
    const options = {
      autoPad: 'explicit',
      padding: [0, 1, 0, 1],
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'hwoi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d autopad explicit nhwc ohwi', async () => {
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
        0., 0., 0., 0., 1., 1., 1., 1., 3., 3., 2., 2., 0., 0., 0., 0., 1., 1.,
        1., 1., 3., 3., 2., 2., 3., 3., 3., 3., 8., 8., 5., 5., 12, 12, 7., 7.,
        3., 3., 3., 3., 7., 7., 4., 4., 9., 9., 5., 5., 9., 9., 9., 9., 20, 20,
        11, 11, 24, 24, 13, 13, 6., 6., 6., 6., 13, 13, 7., 7., 15, 15, 8., 8.,
      ],
    };
    const options = {
      autoPad: 'explicit',
      padding: [0, 1, 0, 1],
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d autopad same-lower nchw iohw', async () => {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [1, 2, 3, 3],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 6, 6, 2],
      data: [
        0, 1, 1, 3, 2, 2,
        3, 8, 5, 12, 7, 7,
        3, 7, 4, 9, 5, 5,
        9, 20, 11, 24, 13, 13,
        6, 13, 7, 15, 8, 8,
        6, 13, 7, 15, 8, 8,
        0, 1, 1, 3, 2, 2,
        3, 8, 5, 12, 7, 7,
        3, 7, 4, 9, 5, 5,
        9, 20, 11, 24, 13, 13,
        6, 13, 7, 15, 8, 8,
        6, 13, 7, 15, 8, 8,
      ],
    };
    const options = {
      autoPad: 'same-lower',
      padding: [0, 1, 0, 1],
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'iohw',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d autopad same-lower nchw hwoi', async () => {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [3, 3, 2, 1],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 6, 6, 2],
      data: [
        0, 1, 1, 3, 2, 2,
        3, 8, 5, 12, 7, 7,
        3, 7, 4, 9, 5, 5,
        9, 20, 11, 24, 13, 13,
        6, 13, 7, 15, 8, 8,
        6, 13, 7, 15, 8, 8,
        0, 1, 1, 3, 2, 2,
        3, 8, 5, 12, 7, 7,
        3, 7, 4, 9, 5, 5,
        9, 20, 11, 24, 13, 13,
        6, 13, 7, 15, 8, 8,
        6, 13, 7, 15, 8, 8,
      ],
    };
    const options = {
      autoPad: 'same-lower',
      padding: [0, 1, 0, 1],
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'hwoi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d autopad same-lower nchw ohwi', async () => {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [2, 3, 3, 1],
      data: new Float32Array(18).fill(1),
    };
    const expected = {
      shape: [1, 6, 6, 2],
      data: [
        0, 1, 1, 3, 2, 2,
        3, 8, 5, 12, 7, 7,
        3, 7, 4, 9, 5, 5,
        9, 20, 11, 24, 13, 13,
        6, 13, 7, 15, 8, 8,
        6, 13, 7, 15, 8, 8,
        0, 1, 1, 3, 2, 2,
        3, 8, 5, 12, 7, 7,
        3, 7, 4, 9, 5, 5,
        9, 20, 11, 24, 13, 13,
        6, 13, 7, 15, 8, 8,
        6, 13, 7, 15, 8, 8,
      ],
    };
    const options = {
      autoPad: 'same-lower',
      padding: [0, 1, 0, 1],
      strides: [2, 2],
      inputLayout: 'nchw',
      filterLayout: 'ohwi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d autopad same-lower nhwc iohw', async () => {
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
        0., 0., 1.,  1.,  1.,  1.,  3.,  3.,  2.,  2.,  2.,  2.,
        3., 3., 8.,  8.,  5.,  5.,  12., 12., 7.,  7.,  7.,  7.,
        3., 3., 7.,  7.,  4.,  4.,  9.,  9.,  5.,  5.,  5.,  5.,
        9., 9., 20., 20., 11., 11., 24., 24., 13., 13., 13., 13.,
        6., 6., 13., 13., 7.,  7.,  15., 15., 8.,  8.,  8.,  8.,
        6., 6., 13., 13., 7.,  7.,  15., 15., 8.,  8.,  8.,  8.,
      ],
    };
    const options = {
      autoPad: 'same-lower',
      padding: [0, 1, 0, 1],
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'iohw',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d autopad same-lower nhwc hwoi', async () => {
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
        0., 0., 1.,  1.,  1.,  1.,  3.,  3.,  2.,  2.,  2.,  2.,
        3., 3., 8.,  8.,  5.,  5.,  12., 12., 7.,  7.,  7.,  7.,
        3., 3., 7.,  7.,  4.,  4.,  9.,  9.,  5.,  5.,  5.,  5.,
        9., 9., 20., 20., 11., 11., 24., 24., 13., 13., 13., 13.,
        6., 6., 13., 13., 7.,  7.,  15., 15., 8.,  8.,  8.,  8.,
        6., 6., 13., 13., 7.,  7.,  15., 15., 8.,  8.,  8.,  8.,
      ],
    };
    const options = {
      autoPad: 'same-lower',
      padding: [0, 1, 0, 1],
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'hwoi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d autopad same-lower nhwc ohwi', async () => {
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
        0., 0., 1.,  1.,  1.,  1.,  3.,  3.,  2.,  2.,  2.,  2.,
        3., 3., 8.,  8.,  5.,  5.,  12., 12., 7.,  7.,  7.,  7.,
        3., 3., 7.,  7.,  4.,  4.,  9.,  9.,  5.,  5.,  5.,  5.,
        9., 9., 20., 20., 11., 11., 24., 24., 13., 13., 13., 13.,
        6., 6., 13., 13., 7.,  7.,  15., 15., 8.,  8.,  8.,  8.,
        6., 6., 13., 13., 7.,  7.,  15., 15., 8.,  8.,  8.,  8.,
      ],
    };
    const options = {
      autoPad: 'same-lower',
      padding: [0, 1, 0, 1],
      strides: [2, 2],
      inputLayout: 'nhwc',
      filterLayout: 'ohwi',
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d outputSizes ignored output padding', async () => {
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
      outputSizes: [10, 8],
    };
    await testConvTranspose2d(input, filter, expected, options);
  });

  it('convTranspose2d bias nchw iohw', async () => {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [1, 2, 3, 3],
      data: new Float32Array(18).fill(1),
    };
    const bias = {
      shape: [1],
      data: new Float32Array([1]),
    };
    const options = {
      inputLayout: 'nchw',
      filterLayout: 'iohw',
    };
    const expected = {
      shape: [1, 2, 5, 5],
      data: [
        1, 2, 4, 4, 3, 4, 9, 16, 13, 8,
        10, 22, 37, 28, 16, 10, 21, 34, 25, 14,
        7, 14, 22, 16, 9, 1, 2, 4, 4, 3,
        4, 9, 16, 13, 8, 10, 22, 37, 28, 16,
        10, 21, 34, 25, 14, 7, 14, 22, 16, 9,
      ],
    };
    await testConvTranspose2d(input, filter, expected, options, bias);
  });

  it('convTranspose2d activation nchw iohw', async () => {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [1, 2, 3, 3],
      data: new Float32Array(18).fill(1),
    };
    const options = {
      inputLayout: 'nchw',
      filterLayout: 'iohw',
    };
    const expected = {
      shape: [1, 2, 5, 5],
      data: [
        0, 1, 3, 3, 2, 3, 8, 15, 12, 7,
        9, 21, 36, 27, 15, 9, 20, 33, 24, 13,
        6, 13, 21, 15, 8, 0, 1, 3, 3, 2,
        3, 8, 15, 12, 7, 9, 21, 36, 27, 15,
        9, 20, 33, 24, 13, 6, 13, 21, 15, 8,
      ],
    };
    await testConvTranspose2d(
        input, filter, expected, options, undefined, 'relu');
  });

  it('convTranspose2d bias activation nchw iohw', async () => {
    const input = {
      shape: [1, 1, 3, 3],
      data: new Float32Array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
    };
    const filter = {
      shape: [1, 2, 3, 3],
      data: new Float32Array(18).fill(1),
    };
    const activation = 'relu6';
    const bias = {
      shape: [1],
      data: new Float32Array([1]),
    };
    const options = {
      inputLayout: 'nchw',
      filterLayout: 'iohw',
    };
    const expected = {
      shape: [1, 2, 5, 5],
      data: [
        1, 2, 4, 4, 3, 4, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 1, 2, 4, 4, 3,
        4, 6, 6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
      ],
    };
    await testConvTranspose2d(
        input, filter, expected, options, bias, activation);
  });
});
