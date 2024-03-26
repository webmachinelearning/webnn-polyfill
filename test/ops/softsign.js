'use strict';
import * as utils from '../utils.js';

describe('test softsign', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('softsign', async () => {
    const builder = new MLGraphBuilder(context);
    const x = builder.input('x', {dataType: 'float32', dimensions: [3]});
    const y = builder.softsign(x);
    utils.checkDataType(y.dataType(), x.dataType());
    utils.checkShape(y.shape(), [3]);
    const graph = await builder.build({y});
    const inputs = {
      'x': new Float32Array([-1, 0, 1]),
    };
    const outputs = {'y': new Float32Array(utils.sizeOfShape([3]))};
    const result = await context.compute(graph, inputs, outputs);
    const expected = [-0.5, 0, 0.5];
    utils.checkValue(result.outputs.y, expected);
  });
});
