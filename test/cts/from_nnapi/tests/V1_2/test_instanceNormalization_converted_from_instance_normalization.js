'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('test instanceNormalization converted from instance_normalization_nhwc test', async () => {
    // Converted test case (from: V1_2/instance_normalization.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {type: 'float32', dimensions: [2, 2, 2, 2]});
    const inputData = new Float32Array([0, 0, 0, -2, 0, -2, 0, 4, 1, -1, -1, 2, -1, -2, 1, 1]);
    const param = builder.constant({type: 'float32', dimensions: [2]}, new Float32Array([1.0, 1.0]));
    const param1 = builder.constant({type: 'float32', dimensions: [2]}, new Float32Array([0.0, 0.0]));
    const param2 = 0.0001;
    const layout = 'nhwc';
    const expected = [0.0, 0.0, 0.0, -0.8164898, 0.0, -0.8164898, 0.0, 1.6329796, 0.99995005, -0.6324429, -0.99995005, 1.2648858, -0.99995005, -1.2648858, 0.99995005, 0.6324429];
    const out = builder.instanceNormalization(input, {'scale': param, 'bias': param1, 'epsilon': param2, 'layout': layout});
    const graph = await builder.build({out});
    const outputs = {out: new Float32Array(utils.sizeOfShape([2, 2, 2, 2]))};
    await context.compute(graph, {'input': inputData}, outputs);
    utils.checkValue(outputs.out, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test instanceNormalization converted from instance_normalization_nhwc_relaxed test', async () => {
    // Converted test case (from: V1_2/instance_normalization.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {type: 'float32', dimensions: [2, 2, 2, 2]});
    const inputData = new Float32Array([0, 0, 0, -2, 0, -2, 0, 4, 1, -1, -1, 2, -1, -2, 1, 1]);
    const param = builder.constant({type: 'float32', dimensions: [2]}, new Float32Array([1.0, 1.0]));
    const param1 = builder.constant({type: 'float32', dimensions: [2]}, new Float32Array([0.0, 0.0]));
    const param2 = 0.0001;
    const layout = 'nhwc';
    const expected = [0.0, 0.0, 0.0, -0.8164898, 0.0, -0.8164898, 0.0, 1.6329796, 0.99995005, -0.6324429, -0.99995005, 1.2648858, -0.99995005, -1.2648858, 0.99995005, 0.6324429];
    const out = builder.instanceNormalization(input, {'scale': param, 'bias': param1, 'epsilon': param2, 'layout': layout});
    const graph = await builder.build({out});
    const outputs = {out: new Float32Array(utils.sizeOfShape([2, 2, 2, 2]))};
    await context.compute(graph, {'input': inputData}, outputs);
    utils.checkValue(outputs.out, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test instanceNormalization converted from instance_normalization_nchw test', async () => {
    // Converted test case (from: V1_2/instance_normalization.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {type: 'float32', dimensions: [2, 2, 2, 2]});
    const inputData = new Float32Array([0, 0, 0, 0, 0, -2, -2, 4, 1, -1, -1, 1, -1, 2, -2, 1]);
    const param = builder.constant({type: 'float32', dimensions: [2]}, new Float32Array([1.0, 1.0]));
    const param1 = builder.constant({type: 'float32', dimensions: [2]}, new Float32Array([0.0, 0.0]));
    const param2 = 0.0001;
    const layout = 'nchw';
    const expected = [0.0, 0.0, 0.0, 0.0, 0.0, -0.8164898, -0.8164898, 1.6329796, 0.99995005, -0.99995005, -0.99995005, 0.99995005, -0.6324429, 1.2648858, -1.2648858, 0.6324429];
    const out = builder.instanceNormalization(input, {'scale': param, 'bias': param1, 'epsilon': param2, 'layout': layout});
    const graph = await builder.build({out});
    const outputs = {out: new Float32Array(utils.sizeOfShape([2, 2, 2, 2]))};
    await context.compute(graph, {'input': inputData}, outputs);
    utils.checkValue(outputs.out, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test instanceNormalization converted from instance_normalization_nchw_relaxed test', async () => {
    // Converted test case (from: V1_2/instance_normalization.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {type: 'float32', dimensions: [2, 2, 2, 2]});
    const inputData = new Float32Array([0, 0, 0, 0, 0, -2, -2, 4, 1, -1, -1, 1, -1, 2, -2, 1]);
    const param = builder.constant({type: 'float32', dimensions: [2]}, new Float32Array([1.0, 1.0]));
    const param1 = builder.constant({type: 'float32', dimensions: [2]}, new Float32Array([0.0, 0.0]));
    const param2 = 0.0001;
    const layout = 'nchw';
    const expected = [0.0, 0.0, 0.0, 0.0, 0.0, -0.8164898, -0.8164898, 1.6329796, 0.99995005, -0.99995005, -0.99995005, 0.99995005, -0.6324429, 1.2648858, -1.2648858, 0.6324429];
    const out = builder.instanceNormalization(input, {'scale': param, 'bias': param1, 'epsilon': param2, 'layout': layout});
    const graph = await builder.build({out});
    const outputs = {out: new Float32Array(utils.sizeOfShape([2, 2, 2, 2]))};
    await context.compute(graph, {'input': inputData}, outputs);
    utils.checkValue(outputs.out, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test instanceNormalization converted from instance_normalization_nhwc_2 test', async () => {
    // Converted test case (from: V1_2/instance_normalization.mod.py)
    const builder = new MLGraphBuilder(context);
    const input1 = builder.input('input1', {type: 'float32', dimensions: [2, 2, 2, 2]});
    const input1Data = new Float32Array([0, 0, 0, -2, 0, -2, 0, 4, 1, -1, -1, 2, -1, -2, 1, 1]);
    const param3 = builder.constant({type: 'float32', dimensions: [2]}, new Float32Array([2.0, 2.0]));
    const param4 = builder.constant({type: 'float32', dimensions: [2]}, new Float32Array([10.0, 10.0]));
    const param5 = 0.0001;
    const layout = 'nhwc';
    const expected = [10.0, 10.0, 10.0, 8.367021, 10.0, 8.367021, 10.0, 13.265959, 11.9999, 8.735114, 8.0001, 12.529772, 8.0001, 7.470228, 11.9999, 11.264886];
    const out1 = builder.instanceNormalization(input1, {'scale': param3, 'bias': param4, 'epsilon': param5, 'layout': layout});
    const graph = await builder.build({out1});
    const outputs = {out1: new Float32Array(utils.sizeOfShape([2, 2, 2, 2]))};
    await context.compute(graph, {'input1': input1Data}, outputs);
    utils.checkValue(outputs.out1, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test instanceNormalization converted from instance_normalization_nhwc_relaxed_2 test', async () => {
    // Converted test case (from: V1_2/instance_normalization.mod.py)
    const builder = new MLGraphBuilder(context);
    const input1 = builder.input('input1', {type: 'float32', dimensions: [2, 2, 2, 2]});
    const input1Data = new Float32Array([0, 0, 0, -2, 0, -2, 0, 4, 1, -1, -1, 2, -1, -2, 1, 1]);
    const param3 = builder.constant({type: 'float32', dimensions: [2]}, new Float32Array([2.0, 2.0]));
    const param4 = builder.constant({type: 'float32', dimensions: [2]}, new Float32Array([10.0, 10.0]));
    const param5 = 0.0001;
    const layout = 'nhwc';
    const expected = [10.0, 10.0, 10.0, 8.367021, 10.0, 8.367021, 10.0, 13.265959, 11.9999, 8.735114, 8.0001, 12.529772, 8.0001, 7.470228, 11.9999, 11.264886];
    const out1 = builder.instanceNormalization(input1, {'scale': param3, 'bias': param4, 'epsilon': param5, 'layout': layout});
    const graph = await builder.build({out1});
    const outputs = {out1: new Float32Array(utils.sizeOfShape([2, 2, 2, 2]))};
    await context.compute(graph, {'input1': input1Data}, outputs);
    utils.checkValue(outputs.out1, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test instanceNormalization converted from instance_normalization_nchw_2 test', async () => {
    // Converted test case (from: V1_2/instance_normalization.mod.py)
    const builder = new MLGraphBuilder(context);
    const input1 = builder.input('input1', {type: 'float32', dimensions: [2, 2, 2, 2]});
    const input1Data = new Float32Array([0, 0, 0, 0, 0, -2, -2, 4, 1, -1, -1, 1, -1, 2, -2, 1]);
    const param3 = builder.constant({type: 'float32', dimensions: [2]}, new Float32Array([2.0, 2.0]));
    const param4 = builder.constant({type: 'float32', dimensions: [2]}, new Float32Array([10.0, 10.0]));
    const param5 = 0.0001;
    const layout = 'nchw';
    const expected = [10.0, 10.0, 10.0, 10.0, 10.0, 8.367021, 8.367021, 13.265959, 11.9999, 8.0001, 8.0001, 11.9999, 8.735114, 12.529772, 7.470228, 11.264886];
    const out1 = builder.instanceNormalization(input1, {'scale': param3, 'bias': param4, 'epsilon': param5, 'layout': layout});
    const graph = await builder.build({out1});
    const outputs = {out1: new Float32Array(utils.sizeOfShape([2, 2, 2, 2]))};
    await context.compute(graph, {'input1': input1Data}, outputs);
    utils.checkValue(outputs.out1, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test instanceNormalization converted from instance_normalization_nchw_relaxed_2 test', async () => {
    // Converted test case (from: V1_2/instance_normalization.mod.py)
    const builder = new MLGraphBuilder(context);
    const input1 = builder.input('input1', {type: 'float32', dimensions: [2, 2, 2, 2]});
    const input1Data = new Float32Array([0, 0, 0, 0, 0, -2, -2, 4, 1, -1, -1, 1, -1, 2, -2, 1]);
    const param3 = builder.constant({type: 'float32', dimensions: [2]}, new Float32Array([2.0, 2.0]));
    const param4 = builder.constant({type: 'float32', dimensions: [2]}, new Float32Array([10.0, 10.0]));
    const param5 = 0.0001;
    const layout = 'nchw';
    const expected = [10.0, 10.0, 10.0, 10.0, 10.0, 8.367021, 8.367021, 13.265959, 11.9999, 8.0001, 8.0001, 11.9999, 8.735114, 12.529772, 7.470228, 11.264886];
    const out1 = builder.instanceNormalization(input1, {'scale': param3, 'bias': param4, 'epsilon': param5, 'layout': layout});
    const graph = await builder.build({out1});
    const outputs = {out1: new Float32Array(utils.sizeOfShape([2, 2, 2, 2]))};
    await context.compute(graph, {'input1': input1Data}, outputs);
    utils.checkValue(outputs.out1, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });
});
/* eslint-disable max-len */
