'use strict';
import * as utils from '../utils.js';

describe('test softmax', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('softmax', async () => {
    const builder = new MLGraphBuilder(context);
    const x = builder.input('x', {type: 'float32', dimensions: [3, 4]});
    const y = builder.softmax(x);
    const graph = await builder.build({y});
    const inputs = {
      'x': new Float32Array([
        0.4301911,
        0.54719144,
        -1.1637765,
        0.18390046,
        0.58390397,
        0.1735679,
        0.539724,
        -0.953514,
        -0.59202826,
        -0.17344485,
        0.14395015,
        -0.37920907,
      ]),
    };
    const outputs = {'y': new Float32Array(utils.sizeOfShape([3, 4]))};
    await context.compute(graph, inputs, outputs);
    const expected = [
      0.32165375,
      0.36157736,
      0.0653337,
      0.25143513,
      0.35271573,
      0.23400122,
      0.33747196,
      0.07581109,
      0.17110129,
      0.26004094,
      0.35717794,
      0.21167983,
    ];
    utils.checkValue(outputs.y, expected);
  });
});
