'use strict';
import * as utils from '../utils.js';

describe('test pow', function() {
  const nn = navigator.ml.getNeuralNetworkContext();
  async function testSqrt(input, expected, shape) {
    const builder = nn.createModelBuilder();
    const x = builder.input('x', {type: 'float32', dimensions: shape});
    const y = builder.constant(
        {type: 'float32', dimensions: [1]}, new Float32Array([0.5]));
    const z = builder.pow(x, y);
    const model = builder.createModel({z});
    const compiledModel = await model.compile();
    const inputs = {'x': {buffer: new Float32Array(input)}};
    const outputs = await compiledModel.compute(inputs);
    utils.checkShape(outputs.z.dimensions, shape);
    utils.checkValue(outputs.z.buffer, expected);
  }
  it('sqrt 1d', async function() {
    await testSqrt([1, 4, 9], [1, 2, 3], [3]);
  });

  it('sqrt 3d', async function() {
    await testSqrt(
        [
          0.33435354, 0.57139647, 0.03689031, 0.7820907,  0.7718887,
          0.17709309, 1.05624,    2.2693596,  1.0328789,  1.6043026,
          2.0692635,  1.7839943,  1.4888871,  0.57544494, 0.2760935,
          0.25916228, 0.24607088, 0.75507194, 0.9365655,  0.66641825,
          0.1919839,  0.42336762, 1.1776822,  1.8486708,  0.7361624,
          0.28052628, 0.261271,   1.0593715,  0.54762685, 0.61064255,
          0.6917134,  0.3692974,  0.01287235, 0.6559981,  0.32968605,
          1.9361054,  1.5982035,  0.49353063, 0.28142217, 0.55740887,
          0.43017766, 2.6145968,  0.4801058,  0.7487864,  1.0473998,
          0.11505236, 0.24899477, 0.21978393, 0.21973193, 0.6550839,
          0.7919175,  0.21990986, 0.2881369,  0.5660939,  0.54675615,
          0.70638055, 0.82219034, 0.6266006,  0.89149487, 0.36557788,
        ],
        [
          0.5782331,  0.7559077,  0.1920685,  0.88435894, 0.8785719,
          0.4208243,  1.0277354,  1.5064393,  1.0163065,  1.2666107,
          1.4384935,  1.3356625,  1.2201996,  0.75858086, 0.525446,
          0.5090798,  0.4960553,  0.86894876, 0.9677631,  0.81634444,
          0.43815967, 0.6506671,  1.0852107,  1.3596584,  0.8579991,
          0.5296473,  0.5111467,  1.0292578,  0.7400181,  0.7814362,
          0.8316931,  0.60769844, 0.11345637, 0.8099371,  0.5741829,
          1.39144,    1.2642008,  0.70251733, 0.53049237, 0.7465982,
          0.6558793,  1.6169715,  0.69289666, 0.86532444, 1.0234255,
          0.3391937,  0.49899375, 0.46881118, 0.46875572, 0.80937254,
          0.88989747, 0.46894547, 0.5367839,  0.7523921,  0.7394296,
          0.8404645,  0.9067471,  0.7915811,  0.9441901,  0.60463035,
        ],
        [3, 4, 5]);
  });

  it('pow 1d', async function() {
    const builder = nn.createModelBuilder();
    const x = builder.input('x', {type: 'float32', dimensions: [3]});
    const y = builder.constant(
        {type: 'float32', dimensions: [3]}, new Float32Array([4, 5, 6]));
    const z = builder.pow(x, y);
    const model = builder.createModel({z});
    const compiledModel = await model.compile();
    const inputs = {'x': {buffer: new Float32Array([1, 2, 3])}};
    const outputs = await compiledModel.compute(inputs);
    utils.checkShape(outputs.z.dimensions, [3]);
    utils.checkValue(outputs.z.buffer, [1., 32., 729.]);
  });

  it('pow broadcast scalar', async function() {
    const builder = nn.createModelBuilder();
    const x = builder.input('x', {type: 'float32', dimensions: [3]});
    const y = builder.constant(
        {type: 'float32', dimensions: [1]}, new Float32Array([2]));
    const z = builder.pow(x, y);
    const model = builder.createModel({z});
    const compiledModel = await model.compile();
    const inputs = {'x': {buffer: new Float32Array([1, 2, 3])}};
    const outputs = await compiledModel.compute(inputs);
    utils.checkShape(outputs.z.dimensions, [3]);
    utils.checkValue(outputs.z.buffer, [1., 4., 9.]);
  });

  it('pow broadcast scalar', async function() {
    const builder = nn.createModelBuilder();
    const x = builder.input('x', {type: 'float32', dimensions: [2, 3]});
    const y = builder.constant(
        {type: 'float32', dimensions: [3]}, new Float32Array([1, 2, 3]));
    const z = builder.pow(x, y);
    const model = builder.createModel({z});
    const compiledModel = await model.compile();
    const inputs = {'x': {buffer: new Float32Array([1, 2, 3, 4, 5, 6])}};
    const outputs = await compiledModel.compute(inputs);
    utils.checkShape(outputs.z.dimensions, [2, 3]);
    utils.checkValue(outputs.z.buffer, [1., 4., 27., 4., 25., 216.]);
  });
});
