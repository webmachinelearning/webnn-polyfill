'use strict';
import * as utils from '../utils.js';

describe('test squeeze', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  async function testSqueeze(oldShape, axes, expectedShape) {
    const builder = new MLGraphBuilder(context);
    const x = builder.input('x', {type: 'float32', dimensions: oldShape});
    const y = builder.squeeze(x, {axes});
    const graph = await builder.build({y});
    const bufferSize = utils.sizeOfShape(oldShape);
    const inputBuffer = new Float32Array(bufferSize);
    for (let i = 0; i < inputBuffer.length; ++i) {
      inputBuffer[i] = Math.random();
    }
    const inputs = {'x': inputBuffer};
    const outputs = {'y': new Float32Array(utils.sizeOfShape(expectedShape))};
    await context.compute(graph, inputs, outputs);
    utils.checkValue(outputs.y, inputBuffer);
  }

  it('squeeze one dimension by default', async () => {
    await testSqueeze([1, 3, 4, 5], undefined, [3, 4, 5]);
  });

  it('squeeze one dimension with axes', async () => {
    await testSqueeze([1, 3, 1, 5], [0], [3, 1, 5]);
  });

  it('squeeze two dimensions by default', async () => {
    await testSqueeze([1, 3, 1, 5], undefined, [3, 5]);
  });

  it('squeeze two dimensions with axes', async () => {
    await testSqueeze([1, 3, 1, 5], [0, 2], [3, 5]);
  });
});
