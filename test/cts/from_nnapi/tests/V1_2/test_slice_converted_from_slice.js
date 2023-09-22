'use strict';
import * as utils from '../../../../utils.js';

/* eslint-disable max-len */
describe('CTS converted from NNAPI CTS', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  it('test slice converted from slice test', async () => {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {dataType: 'float32', dimensions: [4]});
    const inputData = new Float32Array([1, 2, 3, 4]);
    const begin = [1];
    const size = [2];
    const expected = [2, 3];
    const output = builder.slice(input, begin, size);
    const graph = await builder.build({output});
    const outputs = {output: new Float32Array(utils.sizeOfShape([2]))};
    const computeResult = await context.compute(graph, {'input': inputData}, outputs);
    utils.checkValue(computeResult.outputs.output, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test slice converted from slice_relaxed test', async () => {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input = builder.input('input', {dataType: 'float32', dimensions: [4]});
    const inputData = new Float32Array([1, 2, 3, 4]);
    const begin = [1];
    const size = [2];
    const expected = [2, 3];
    const output = builder.slice(input, begin, size);
    const graph = await builder.build({output});
    const outputs = {output: new Float32Array(utils.sizeOfShape([2]))};
    const computeResult = await context.compute(graph, {'input': inputData}, outputs);
    utils.checkValue(computeResult.outputs.output, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test slice converted from slice_2 test', async () => {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input1 = builder.input('input1', {dataType: 'float32', dimensions: [2, 3]});
    const input1Data = new Float32Array([1, 2, 3, 4, 5, 6]);
    const begin1 = [1, 0];
    const size1 = [1, 2];
    const expected = [4, 5];
    const output1 = builder.slice(input1, begin1, size1);
    const graph = await builder.build({output1});
    const outputs = {output1: new Float32Array(utils.sizeOfShape([1, 2]))};
    const computeResult = await context.compute(graph, {'input1': input1Data}, outputs);
    utils.checkValue(computeResult.outputs.output1, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test slice converted from slice_relaxed_2 test', async () => {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input1 = builder.input('input1', {dataType: 'float32', dimensions: [2, 3]});
    const input1Data = new Float32Array([1, 2, 3, 4, 5, 6]);
    const begin1 = [1, 0];
    const size1 = [1, 2];
    const expected = [4, 5];
    const output1 = builder.slice(input1, begin1, size1);
    const graph = await builder.build({output1});
    const outputs = {output1: new Float32Array(utils.sizeOfShape([1, 2]))};
    const computeResult = await context.compute(graph, {'input1': input1Data}, outputs);
    utils.checkValue(computeResult.outputs.output1, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test slice converted from slice_3 test', async () => {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input2 = builder.input('input2', {dataType: 'float32', dimensions: [2, 3, 2]});
    const input2Data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const begin2 = [0, 0, 0];
    const size2 = [2, 3, 2];
    const expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const output2 = builder.slice(input2, begin2, size2);
    const graph = await builder.build({output2});
    const outputs = {output2: new Float32Array(utils.sizeOfShape([2, 3, 2]))};
    const computeResult = await context.compute(graph, {'input2': input2Data}, outputs);
    utils.checkValue(computeResult.outputs.output2, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test slice converted from slice_relaxed_3 test', async () => {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input2 = builder.input('input2', {dataType: 'float32', dimensions: [2, 3, 2]});
    const input2Data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    const begin2 = [0, 0, 0];
    const size2 = [2, 3, 2];
    const expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    const output2 = builder.slice(input2, begin2, size2);
    const graph = await builder.build({output2});
    const outputs = {output2: new Float32Array(utils.sizeOfShape([2, 3, 2]))};
    const computeResult = await context.compute(graph, {'input2': input2Data}, outputs);
    utils.checkValue(computeResult.outputs.output2, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test slice converted from slice_4 test', async () => {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input3 = builder.input('input3', {dataType: 'float32', dimensions: [4, 1, 1, 1]});
    const input3Data = new Float32Array([1, 2, 3, 4]);
    const begin3 = [1, 0, 0, 0];
    const size3 = [3, 1, 1, 1];
    const expected = [2, 3, 4];
    const output3 = builder.slice(input3, begin3, size3);
    const graph = await builder.build({output3});
    const outputs = {output3: new Float32Array(utils.sizeOfShape([3, 1, 1, 1]))};
    const computeResult = await context.compute(graph, {'input3': input3Data}, outputs);
    utils.checkValue(computeResult.outputs.output3, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test slice converted from slice_relaxed_4 test', async () => {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input3 = builder.input('input3', {dataType: 'float32', dimensions: [4, 1, 1, 1]});
    const input3Data = new Float32Array([1, 2, 3, 4]);
    const begin3 = [1, 0, 0, 0];
    const size3 = [3, 1, 1, 1];
    const expected = [2, 3, 4];
    const output3 = builder.slice(input3, begin3, size3);
    const graph = await builder.build({output3});
    const outputs = {output3: new Float32Array(utils.sizeOfShape([3, 1, 1, 1]))};
    const computeResult = await context.compute(graph, {'input3': input3Data}, outputs);
    utils.checkValue(computeResult.outputs.output3, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test slice converted from slice_5 test', async () => {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input4 = builder.input('input4', {dataType: 'int32', dimensions: [3, 2, 3, 1]});
    const input4Data = new Int32Array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]);
    const begin4 = [1, 0, 0, 0];
    const size4 = [1, 1, 3, 1];
    const expected = [3, 3, 3];
    const output4 = builder.slice(input4, begin4, size4);
    const graph = await builder.build({output4});
    const outputs = {output4: new Int32Array(utils.sizeOfShape([1, 1, 3, 1]))};
    const computeResult = await context.compute(graph, {'input4': input4Data}, outputs);
    utils.checkValue(computeResult.outputs.output4, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test slice converted from slice_relaxed_5 test', async () => {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input4 = builder.input('input4', {dataType: 'int32', dimensions: [3, 2, 3, 1]});
    const input4Data = new Int32Array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]);
    const begin4 = [1, 0, 0, 0];
    const size4 = [1, 1, 3, 1];
    const expected = [3, 3, 3];
    const output4 = builder.slice(input4, begin4, size4);
    const graph = await builder.build({output4});
    const outputs = {output4: new Int32Array(utils.sizeOfShape([1, 1, 3, 1]))};
    const computeResult = await context.compute(graph, {'input4': input4Data}, outputs);
    utils.checkValue(computeResult.outputs.output4, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test slice converted from slice_float16_5 test', async () => {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input4 = builder.input('input4', {dataType: 'int32', dimensions: [3, 2, 3, 1]});
    const input4Data = new Int32Array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]);
    const begin4 = [1, 0, 0, 0];
    const size4 = [1, 1, 3, 1];
    const expected = [3, 3, 3];
    const output4 = builder.slice(input4, begin4, size4);
    const graph = await builder.build({output4});
    const outputs = {output4: new Int32Array(utils.sizeOfShape([1, 1, 3, 1]))};
    const computeResult = await context.compute(graph, {'input4': input4Data}, outputs);
    utils.checkValue(computeResult.outputs.output4, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test slice converted from slice_6 test', async () => {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input5 = builder.input('input5', {dataType: 'int32', dimensions: [3, 2, 3, 1]});
    const input5Data = new Int32Array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]);
    const begin5 = [1, 0, 0, 0];
    const size5 = [2, 1, 3, 1];
    const expected = [3, 3, 3, 5, 5, 5];
    const output5 = builder.slice(input5, begin5, size5);
    const graph = await builder.build({output5});
    const outputs = {output5: new Int32Array(utils.sizeOfShape([2, 1, 3, 1]))};
    const computeResult = await context.compute(graph, {'input5': input5Data}, outputs);
    utils.checkValue(computeResult.outputs.output5, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test slice converted from slice_relaxed_6 test', async () => {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input5 = builder.input('input5', {dataType: 'int32', dimensions: [3, 2, 3, 1]});
    const input5Data = new Int32Array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]);
    const begin5 = [1, 0, 0, 0];
    const size5 = [2, 1, 3, 1];
    const expected = [3, 3, 3, 5, 5, 5];
    const output5 = builder.slice(input5, begin5, size5);
    const graph = await builder.build({output5});
    const outputs = {output5: new Int32Array(utils.sizeOfShape([2, 1, 3, 1]))};
    const computeResult = await context.compute(graph, {'input5': input5Data}, outputs);
    utils.checkValue(computeResult.outputs.output5, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test slice converted from slice_float16_6 test', async () => {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input5 = builder.input('input5', {dataType: 'int32', dimensions: [3, 2, 3, 1]});
    const input5Data = new Int32Array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]);
    const begin5 = [1, 0, 0, 0];
    const size5 = [2, 1, 3, 1];
    const expected = [3, 3, 3, 5, 5, 5];
    const output5 = builder.slice(input5, begin5, size5);
    const graph = await builder.build({output5});
    const outputs = {output5: new Int32Array(utils.sizeOfShape([2, 1, 3, 1]))};
    const computeResult = await context.compute(graph, {'input5': input5Data}, outputs);
    utils.checkValue(computeResult.outputs.output5, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test slice converted from slice_8 test', async () => {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input7 = builder.input('input7', {dataType: 'int32', dimensions: [3, 2, 3, 1]});
    const input7Data = new Int32Array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]);
    const begin7 = [1, 0, 0, 0];
    const size7 = [2, 1, 3, 1];
    const expected = [3, 3, 3, 5, 5, 5];
    const output7 = builder.slice(input7, begin7, size7);
    const graph = await builder.build({output7});
    const outputs = {output7: new Int32Array(utils.sizeOfShape([2, 1, 3, 1]))};
    const computeResult = await context.compute(graph, {'input7': input7Data}, outputs);
    utils.checkValue(computeResult.outputs.output7, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });

  it('test slice converted from slice_relaxed_8 test', async () => {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input7 = builder.input('input7', {dataType: 'int32', dimensions: [3, 2, 3, 1]});
    const input7Data = new Int32Array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]);
    const begin7 = [1, 0, 0, 0];
    const size7 = [2, 1, 3, 1];
    const expected = [3, 3, 3, 5, 5, 5];
    const output7 = builder.slice(input7, begin7, size7);
    const graph = await builder.build({output7});
    const outputs = {output7: new Int32Array(utils.sizeOfShape([2, 1, 3, 1]))};
    const computeResult = await context.compute(graph, {'input7': input7Data}, outputs);
    utils.checkValue(computeResult.outputs.output7, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  });

  it('test slice converted from slice_float16_8 test', async () => {
    // Converted test case (from: V1_2/slice.mod.py)
    const builder = new MLGraphBuilder(context);
    const input7 = builder.input('input7', {dataType: 'int32', dimensions: [3, 2, 3, 1]});
    const input7Data = new Int32Array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]);
    const begin7 = [1, 0, 0, 0];
    const size7 = [2, 1, 3, 1];
    const expected = [3, 3, 3, 5, 5, 5];
    const output7 = builder.slice(input7, begin7, size7);
    const graph = await builder.build({output7});
    const outputs = {output7: new Int32Array(utils.sizeOfShape([2, 1, 3, 1]))};
    const computeResult = await context.compute(graph, {'input7': input7Data}, outputs);
    utils.checkValue(computeResult.outputs.output7, expected, utils.ctsFp32RestrictAccuracyCriteria);
  });
});
/* eslint-disable max-len */
