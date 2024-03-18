'use strict';
import * as utils from '../utils.js';

describe('test pad', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  async function testPad(
      input, beginningPadding, endingPadding, options, expected) {
    const builder = new MLGraphBuilder(context);
    const x =
        builder.input('x', {dataType: 'float32', dimensions: input.shape});
    const y = builder.pad(x, beginningPadding, endingPadding, options);
    utils.checkDataType(y.dataType(), x.dataType());
    utils.checkShape(y.shape(), expected.shape);
    const graph = await builder.build({y});
    const inputs = {'x': new Float32Array(input.values)};
    const outputs = {'y': new Float32Array(utils.sizeOfShape(expected.shape))};
    const result = await context.compute(graph, inputs, outputs);
    utils.checkValue(result.outputs.y, expected.values);
  }

  it('pad default', async () => {
    await testPad(
        {
          shape: [2, 3],
          values: [1, 2, 3, 4, 5, 6],
        },
        [1, 2],
        [1, 2],
        {}, {
          shape: [4, 7],
          values: [
            0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3., 0., 0.,
            0., 0., 4., 5., 6., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          ],
        });
  });

  it('pad constant model default value', async () => {
    await testPad(
        {
          shape: [2, 3],
          values: [1, 2, 3, 4, 5, 6],
        },
        [1, 2],
        [1, 2],
        {mode: 'constant'}, {
          shape: [4, 7],
          values: [
            0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 2., 3., 0., 0.,
            0., 0., 4., 5., 6., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          ],
        });
  });

  it('pad constant model specified value', async () => {
    await testPad(
        {
          shape: [2, 3],
          values: [1, 2, 3, 4, 5, 6],
        },
        [1, 2],
        [1, 2],
        {mode: 'constant',
          value: 9.}, {
          shape: [4, 7],
          values: [
            9., 9., 9., 9., 9., 9., 9., 9., 9., 1., 2., 3., 9., 9.,
            9., 9., 4., 5., 6., 9., 9., 9., 9., 9., 9., 9., 9., 9.,
          ],
        });
  });

  it('pad edge mode', async () => {
    await testPad(
        {
          shape: [2, 3],
          values: [1, 2, 3, 4, 5, 6],
        },
        [1, 2],
        [1, 2],
        {mode: 'edge'}, {
          shape: [4, 7],
          values: [
            1., 1., 1., 2., 3., 3., 3., 1., 1., 1., 2., 3., 3., 3.,
            4., 4., 4., 5., 6., 6., 6., 4., 4., 4., 5., 6., 6., 6.,
          ],
        });
  });

  it('pad reflection mode', async () => {
    await testPad(
        {
          shape: [2, 3],
          values: [1, 2, 3, 4, 5, 6],
        },
        [1, 2],
        [1, 2],
        {mode: 'reflection'}, {
          shape: [4, 7],
          values: [
            6., 5., 4., 5., 6., 5., 4., 3., 2., 1., 2., 3., 2., 1.,
            6., 5., 4., 5., 6., 5., 4., 3., 2., 1., 2., 3., 2., 1.,
          ],
        });
  });

  it('pad symmetric mode', async () => {
    await testPad(
        {
          shape: [2, 3],
          values: [1, 2, 3, 4, 5, 6],
        },
        [1, 2],
        [1, 2],
        {mode: 'symmetric'}, {
          shape: [4, 7],
          values: [
            2., 1., 1., 2., 3., 3., 2., 2., 1., 1., 2., 3., 3., 2.,
            5., 4., 4., 5., 6., 6., 5., 5., 4., 4., 5., 6., 6., 5.,
          ],
        });
  });
});
