'use strict';
import * as utils from '../utils.js';

describe('test pad', function() {
  const context = navigator.ml.createContext();

  async function testPad(input, paddings, options, expected) {
    const builder = new MLGraphBuilder(context);
    const x = builder.input('x', {type: 'float32', dimensions: input.shape});
    const padding = builder.constant(
        {type: 'int32', dimensions: paddings.shape},
        new Int32Array(paddings.values));
    const y = builder.pad(x, padding, options);
    const graph = await builder.build({y});
    const inputs = {'x': {data: new Float32Array(input.values)}};
    const outputs = await graph.compute(inputs);
    utils.checkShape(outputs.y.dimensions, expected.shape);
    utils.checkValue(outputs.y.data, expected.values);
  }

  it('pad default', async function() {
    await testPad(
        {
          shape: [2, 3],
          values: [1, 2, 3, 4, 5, 6],
        },
        {
          shape: [2, 2],
          values: [1, 1, 2, 2],
        },
        {}, {
          shape: [4, 7],
          values: [
            0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3., 0., 0.,
            0., 0., 4., 5., 6., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          ],
        });
  });

  it('pad edge mode', async function() {
    await testPad(
        {
          shape: [2, 3],
          values: [1, 2, 3, 4, 5, 6],
        },
        {
          shape: [2, 2],
          values: [1, 1, 2, 2],
        },
        {mode: 'edge'}, {
          shape: [4, 7],
          values: [
            1., 1., 1., 2., 3., 3., 3., 1., 1., 1., 2., 3., 3., 3.,
            4., 4., 4., 5., 6., 6., 6., 4., 4., 4., 5., 6., 6., 6.,
          ],
        });
  });

  it('pad reflection mode', async function() {
    await testPad(
        {
          shape: [2, 3],
          values: [1, 2, 3, 4, 5, 6],
        },
        {
          shape: [2, 2],
          values: [1, 1, 2, 2],
        },
        {mode: 'reflection'}, {
          shape: [4, 7],
          values: [
            6., 5., 4., 5., 6., 5., 4., 3., 2., 1., 2., 3., 2., 1.,
            6., 5., 4., 5., 6., 5., 4., 3., 2., 1., 2., 3., 2., 1.,
          ],
        });
  });

  it('pad symmetric mode', async function() {
    await testPad(
        {
          shape: [2, 3],
          values: [1, 2, 3, 4, 5, 6],
        },
        {
          shape: [2, 2],
          values: [1, 1, 2, 2],
        },
        {mode: 'symmetric'}, {
          shape: [4, 7],
          values: [
            2., 1., 1., 2., 3., 3., 2., 2., 1., 1., 2., 3., 3., 2.,
            5., 4., 4., 5., 6., 6., 5., 5., 4., 4., 5., 6., 6., 5.,
          ],
        });
  });
});
