'use strict';
import * as utils from '../utils.js';

describe('test hardSwish', async () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('hardSwish', async () => {
    const builder = new MLGraphBuilder(context);
    const x = builder.input('x', {type: 'float32', dimensions: [2, 3]});
    const y = builder.hardSwish(x);
    const graph = await builder.build({y});
    const inputs = {
      'x': new Float32Array([
        -4.2, -3.001, -3., 0.6, 2.994, 3.001,
      ]),
    };
    const outputs = {'y': new Float32Array(utils.sizeOfShape([2, 3]))};
    await context.compute(graph, inputs, outputs);
    const expected = [
      0., 0., 0., 0.36, 2.991006, 3.001,
    ];
    utils.checkValue(outputs.y, expected);
  });
});
