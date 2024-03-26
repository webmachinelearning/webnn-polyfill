'use strict';
import * as utils from '../utils.js';

describe('test softplus', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  async function testSoftplus(input, expected, options = {}) {
    const builder = new MLGraphBuilder(context);
    const x =
      builder.input('x', {dataType: 'float32', dimensions: input.shape});
    const y = builder.softplus(x, options);
    utils.checkDataType(y.dataType(), x.dataType());
    utils.checkShape(y.shape(), x.shape());
    const graph = await builder.build({y});
    const inputs = {'x': new Float32Array(input.value)};
    const outputs = {'y': new Float32Array(utils.sizeOfShape(input.shape))};
    const result = await context.compute(graph, inputs, outputs);
    utils.checkValue(result.outputs.y, expected);
  }

  it('softplus default', async () => {
    await testSoftplus(
        {
          shape: [4],
          value: [-1, 0, 1, 2],
        },
        [0.313261688, 0.693147181, 1.313261688, 2.126928011],
    );
  });

  it('softplus', async () => {
    await testSoftplus(
        {
          shape: [4],
          value: [-1, 0, 1, 2],
        },
        [0.063464023, 0.346573591, 1.063464045, 2.009074926],
        {
          steepness: 2,
        },
    );
  });
});
