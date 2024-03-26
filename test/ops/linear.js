'use strict';
import * as utils from '../utils.js';

describe('test linear', async () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  async function testLinear(input, expected, options = {}) {
    const builder = new MLGraphBuilder(context);
    const x =
        builder.input('x', {dataType: 'float32', dimensions: input.shape});
    const y = builder.linear(x, options);
    utils.checkDataType(y.dataType(), x.dataType());
    utils.checkShape(y.shape(), input.shape);
    const graph = await builder.build({y});
    const inputs = {'x': new Float32Array(input.value)};
    const outputs = {'y': new Float32Array(utils.sizeOfShape(input.shape))};
    const result = await context.compute(graph, inputs, outputs);
    utils.checkValue(result.outputs.y, expected);
  }

  it('linear default', async () => {
    await testLinear(
        {
          shape: [3],
          value: [-1, 0, 1],
        },
        [-1, 0, 1],
    );
  });

  it('linear alpha', async () => {
    await testLinear(
        {
          shape: [3],
          value: [-1, 0, 1],
        },
        [-0.25, 0, 0.25],
        {
          alpha: 0.25,
        },
    );
  });

  it('linear beta', async () => {
    await testLinear(
        {
          shape: [3],
          value: [-1, 0, 1],
        },
        [-0.75, 0.25, 1.25],
        {
          beta: 0.25,
        },
    );
  });

  it('linear', async () => {
    await testLinear(
        {
          shape: [3],
          value: [-1, 0, 1],
        },
        [0, 0.25, 0.5],
        {
          alpha: 0.25,
          beta: 0.25,
        },
    );
  });
});
