'use strict';
import * as utils from '../utils.js';

describe('test hardSigmoid', async () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  async function testHardSigmoid(input, expected, options = {}) {
    const builder = new MLGraphBuilder(context);
    const x =
        builder.input('x', {dataType: 'float32', dimensions: input.shape});
    const y = builder.hardSigmoid(x, options);
    utils.checkDataType(y.dataType(), x.dataType());
    utils.checkShape(y.shape(), x.shape());
    const graph = await builder.build({y});
    const inputs = {'x': new Float32Array(input.value)};
    const outputs = {'y': new Float32Array(utils.sizeOfShape(input.shape))};
    const result = await context.compute(graph, inputs, outputs);
    utils.checkValue(result.outputs.y, expected);
  }

  it('hardSigmoid default', async () => {
    await testHardSigmoid(
        {
          shape: [2, 3],
          value: [-1, 0, 1, 2, 3, 4],
        },
        [0.3, 0.5, 0.7, 0.9, 1, 1],
    );
  });

  it('hardSigmoid alpha', async () => {
    await testHardSigmoid(
        {
          shape: [2, 3],
          value: [-1, 0, 1, 2, 3, 4],
        },
        [0.25, 0.5, 0.75, 1, 1, 1],
        {
          alpha: 0.25,
        },
    );
  });

  it('hardSigmoid beta', async () => {
    await testHardSigmoid(
        {
          shape: [2, 3],
          value: [-1, 0, 1, 2, 3, 4],
        },
        [0.05, 0.25, 0.45, 0.65, 0.85, 1],
        {
          beta: 0.25,
        },
    );
  });

  it('hardSigmoid', async () => {
    await testHardSigmoid(
        {
          shape: [2, 3],
          value: [-1, 0, 1, 2, 3, 4],
        },
        [0, 0.25, 0.5, 0.75, 1, 1],
        {
          alpha: 0.25,
          beta: 0.25,
        },
    );
  });
});
