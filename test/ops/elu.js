'use strict';
import * as utils from '../utils.js';

describe('test elu', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  async function testElu(input, expected, options = {}) {
    const builder = new MLGraphBuilder(context);
    const x =
        builder.input('x', {dataType: 'float32', dimensions: input.shape});
    const y = builder.elu(x, options);
    utils.checkDataType(y.dataType(), x.dataType());
    utils.checkShape(y.shape(), input.shape);
    const graph = await builder.build({y});
    const inputs = {'x': new Float32Array(input.value)};
    const outputs = {'y': new Float32Array(utils.sizeOfShape(input.shape))};
    const result = await context.compute(graph, inputs, outputs);
    utils.checkValue(result.outputs.y, expected);
  }

  it('elu', async () => {
    await testElu(
        {shape: [3], value: [-1, 0, 1]}, [-1.264241118, 0, 1], {alpha: 2});
    await testElu(
        {
          shape: [1, 1, 1, 3],
          value: [-1, 0, 1],
        },
        [1.264241118, 0, 1],
        {alpha: -2},
    );
  });

  it('elu default', async () => {
    await testElu(
        {
          shape: [3],
          value: [-1, 0, 1],
        },
        [-0.632120559, 0, 1],
    );
  });
});
