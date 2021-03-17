'use strict';
import * as utils from '../utils.js';

describe('test resample', function() {
  const nn = navigator.ml.getNeuralNetworkContext();

  async function testResample(input, options, expected) {
    const builder = nn.createModelBuilder();
    const x = builder.input('x', {type: 'float32', dimensions: input.shape});
    const y = builder.resample(x, options);
    const model = builder.createModel({y});
    const compiledModel = await model.compile();
    const inputs = {'x': {buffer: new Float32Array(input.values)}};
    const outputs = await compiledModel.compute(inputs);
    utils.checkShape(outputs.y.dimensions, expected.shape);
    utils.checkValue(outputs.y.buffer, expected.values);
  }

  it('resample upsample scales linear', async function() {
    await testResample(
        {
          shape: [1, 1, 2, 2],
          values: [1, 2, 3, 4],
        },
        {
          mode: 'linear',
          scales: [1.0, 1.0, 2.0, 2.0],
        },
        {
          shape: [1, 1, 4, 4],
          values: [
            1.,
            1.25,
            1.75,
            2.,
            1.5,
            1.75,
            2.25,
            2.5,
            2.5,
            2.75,
            3.25,
            3.5,
            3.,
            3.25,
            3.75,
            4.,
          ],
        });
  });

  it('resample upsample sizes linear', async function() {
    await testResample(
        {
          shape: [1, 1, 2, 2],
          values: [1, 2, 3, 4],
        },
        {
          mode: 'linear',
          sizes: [1, 1, 4, 4],
        },
        {
          shape: [1, 1, 4, 4],
          values: [
            1.,
            1.25,
            1.75,
            2.,
            1.5,
            1.75,
            2.25,
            2.5,
            2.5,
            2.75,
            3.25,
            3.5,
            3.,
            3.25,
            3.75,
            4.,
          ],
        });
  });


  it('resample upsample scales nearest', async function() {
    await testResample(
        {
          shape: [1, 1, 2, 2],
          values: [1, 2, 3, 4],
        },
        {
          mode: 'nearest-neighbor',
          scales: [1.0, 1.0, 2.0, 3.0],
        },
        {
          shape: [1, 1, 4, 6],
          values: [
            1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2,
            3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4,
          ],
        });
  });

  it('resample upsample sizes nearest', async function() {
    await testResample(
        {
          shape: [1, 1, 2, 2],
          values: [1, 2, 3, 4],
        },
        {
          mode: 'nearest-neighbor',
          sizes: [1, 1, 4, 6],
        },
        {
          shape: [1, 1, 4, 6],
          values: [
            1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2,
            3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4,
          ],
        });
  });
});
