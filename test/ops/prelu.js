'use strict';
import * as utils from '../utils.js';

describe('test prelu', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('prelu 3d', async () => {
    const builder = new MLGraphBuilder(context);
    const x = builder.input('x', {dataType: 'float32', dimensions: [1, 2, 3]});
    const slope = builder.input(
        'slope', {dataType: 'float32', dimensions: [1, 2, 3]});
    const y = builder.prelu(x, slope);
    utils.checkDataType(y.dataType(), x.dataType());
    utils.checkShape(y.shape(), x.shape());
    const graph = await builder.build({y});
    const inputs = {
      'x': new Float32Array([
        1, -1, 2,
        -2, 3, -3,
      ]),
      'slope': new Float32Array([
        0.125, -0.125, 0.25,
        -0.25, -0.5, 0.5,
      ]),
    };
    const outputs = {'y': new Float32Array(utils.sizeOfShape([1, 2, 3]))};
    const result = await context.compute(graph, inputs, outputs);
    const expected = [
      1, 0.125, 2,
      0.5, 3, -1.5,
    ];
    utils.checkValue(result.outputs.y, expected);
  });

  it('prelu 3d constant slope', async () => {
    const builder = new MLGraphBuilder(context);
    const x = builder.input('x', {dataType: 'float32', dimensions: [1, 2, 3]});
    const slope = builder.constant(
        {dataType: 'float32', dimensions: [1, 2, 3]},
        new Float32Array([
          0.125, -0.125, 0.25,
          -0.25, -0.5, 0.5,
        ]),
    );
    const y = builder.prelu(x, slope);
    utils.checkDataType(y.dataType(), x.dataType());
    utils.checkShape(y.shape(), x.shape());
    const graph = await builder.build({y});
    const inputs = {
      'x': new Float32Array([
        1, -1, 2,
        -2, 3, -3,
      ]),
    };
    const outputs = {'y': new Float32Array(utils.sizeOfShape([1, 2, 3]))};
    const result = await context.compute(graph, inputs, outputs);
    const expected = [
      1, 0.125, 2,
      0.5, 3, -1.5,
    ];
    utils.checkValue(result.outputs.y, expected);
  });

  it('prelu broadcast 3d x 1d', async () => {
    const builder = new MLGraphBuilder(context);
    const x = builder.input('x', {dataType: 'float32', dimensions: [1, 2, 3]});
    const slope = builder.constant(
        {dataType: 'float32', dimensions: [1]},
        new Float32Array([0.125]),
    );
    const y = builder.prelu(x, slope);
    utils.checkDataType(y.dataType(), x.dataType());
    utils.checkShape(y.shape(), x.shape());
    const graph = await builder.build({y});
    const inputs = {
      'x': new Float32Array([
        1, -1, 2,
        -2, 3, -3,
      ]),
    };
    const outputs = {'y': new Float32Array(utils.sizeOfShape([1, 2, 3]))};
    const result = await context.compute(graph, inputs, outputs);
    const expected = [
      1, -0.125, 2,
      -0.25, 3, -0.375,
    ];
    utils.checkValue(result.outputs.y, expected);
  });

  it('prelu broadcast 3d x 2d', async () => {
    const builder = new MLGraphBuilder(context);
    const x = builder.input('x', {dataType: 'float32', dimensions: [1, 2, 3]});
    const slope = builder.constant(
        {dataType: 'float32', dimensions: [1, 3]},
        new Float32Array([
          0.125, -0.25, 0.5,
        ]),
    );
    const y = builder.prelu(x, slope);
    utils.checkDataType(y.dataType(), x.dataType());
    utils.checkShape(y.shape(), x.shape());
    const graph = await builder.build({y});
    const inputs = {
      'x': new Float32Array([
        1, -1, 2,
        -2, 3, -3,
      ]),
    };
    const outputs = {'y': new Float32Array(utils.sizeOfShape([1, 2, 3]))};
    const result = await context.compute(graph, inputs, outputs);
    const expected = [
      1, 0.25, 2,
      -0.25, 3, -1.5,
    ];
    utils.checkValue(result.outputs.y, expected);
  });

  it('prelu broadcast 3d x 3d', async () => {
    const builder = new MLGraphBuilder(context);
    const x = builder.input('x', {dataType: 'float32', dimensions: [1, 2, 3]});
    const slope = builder.constant(
        {dataType: 'float32', dimensions: [1, 2, 1]},
        new Float32Array([
          0.125,
          -0.125,
        ]),
    );
    const y = builder.prelu(x, slope);
    utils.checkDataType(y.dataType(), x.dataType());
    utils.checkShape(y.shape(), x.shape());
    const graph = await builder.build({y});
    const inputs = {
      'x': new Float32Array([
        1, -1, 2,
        -2, 3, -3,
      ]),
    };
    const outputs = {'y': new Float32Array(utils.sizeOfShape([1, 2, 3]))};
    const result = await context.compute(graph, inputs, outputs);
    const expected = [
      1, -0.125, 2,
      0.25, 3, 0.375,
    ];
    utils.checkValue(result.outputs.y, expected);
  });
});
