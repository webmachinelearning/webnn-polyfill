'use strict';
import * as utils from '../utils.js';

const url = import.meta.url;
const assert = chai.assert;
const testDataDir = '../../test-data/models/mobilenetv2_batchnorm_nchw';

describe('test mobilenetv2 batchnorm nchw', function() {
  // eslint-disable-next-line no-invalid-this
  this.timeout(0);
  let context;
  let graph;
  let fusedGraph;
  let beforeNumBytes;
  let beforeNumTensors;
  before(async () => {
    if (typeof _tfengine !== 'undefined') {
      beforeNumBytes = _tfengine.memory().numBytes;
      beforeNumTensors = _tfengine.memory().numTensors;
    }
    context = await navigator.ml.createContext();
    const builder = new MLGraphBuilder(context);
    let fusedBatchNorm = false;

    function buildBatchNorm(input, mean, variance, scale, bias, relu) {
      const options = {scale: scale, bias: bias};
      if (fusedBatchNorm) {
        if (relu) {
          options.activation = builder.relu();
        }
      }
      let batchNorm =
          builder.batchNormalization(input, mean, variance, options);
      if (!fusedBatchNorm) {
        if (relu) {
          batchNorm = builder.relu(batchNorm);
        }
      }
      return batchNorm;
    }

    async function buildConvBatchNorm(
        input, name, subName, relu, options = undefined) {
      const subPrefix = subName !== '' ? '_linearbottleneck' + subName : '';
      let prefix = testDataDir + '/weights/mobilenetv20_features' + subPrefix;
      const weightsName = `${prefix}_conv${name}_weight.npy`;
      const weights =
          await utils.buildConstantFromNpy(builder, new URL(weightsName, url));
      prefix += '_batchnorm' + name;
      const scaleName = prefix + '_gamma.npy';
      const scale =
          await utils.buildConstantFromNpy(builder, new URL(scaleName, url));
      const biasName = prefix + '_beta.npy';
      const bias =
          await utils.buildConstantFromNpy(builder, new URL(biasName, url));
      const meanName = prefix + '_running_mean.npy';
      const mean =
          await utils.buildConstantFromNpy(builder, new URL(meanName, url));
      const varianceName = prefix + '_running_var.npy';
      const variance =
          await utils.buildConstantFromNpy(builder, new URL(varianceName, url));
      const conv = builder.conv2d(input, weights, options);
      return buildBatchNorm(conv, mean, variance, scale, bias, relu);
    }

    async function buildLinearBottleneck(
        input, subName, options, shortcut = true) {
      const batchNorm0 = await buildConvBatchNorm(input, '0', subName, true);
      const batchNorm1 =
          await buildConvBatchNorm(batchNorm0, '1', subName, true, options);
      const batchNorm2 = await buildConvBatchNorm(batchNorm1, '2', subName);

      if (shortcut) {
        return builder.add(input, batchNorm2);
      }
      return batchNorm2;
    }

    async function buildMobileNet() {
      const padding = [1, 1, 1, 1];
      const strides = [2, 2];
      const data = builder.input(
          'input', {type: 'float32', dimensions: [1, 3, 224, 224]});
      const batch0 =
          await buildConvBatchNorm(data, '0', '', true, {strides, padding});
      const bottleneck0 = await buildLinearBottleneck(
          batch0, '0', {padding, groups: 32}, false);
      const bottleneck1 = await buildLinearBottleneck(
          bottleneck0, '1', {strides, padding, groups: 96}, false);
      const bottleneck2 =
          await buildLinearBottleneck(bottleneck1, '2', {padding, groups: 144});
      const bottleneck3 = await buildLinearBottleneck(
          bottleneck2, '3', {strides, padding, groups: 144}, false);
      const bottleneck4 =
          await buildLinearBottleneck(bottleneck3, '4', {padding, groups: 192});
      const bottleneck5 =
          await buildLinearBottleneck(bottleneck4, '5', {padding, groups: 192});
      const bottleneck6 = await buildLinearBottleneck(
          bottleneck5, '6', {padding, groups: 192}, false);
      const bottleneck7 =
          await buildLinearBottleneck(bottleneck6, '7', {padding, groups: 384});
      const bottleneck8 =
          await buildLinearBottleneck(bottleneck7, '8', {padding, groups: 384});
      const bottleneck9 =
          await buildLinearBottleneck(bottleneck8, '9', {padding, groups: 384});
      const bottleneck10 = await buildLinearBottleneck(
          bottleneck9, '10', {strides, padding, groups: 384}, false);
      const bottleneck11 = await buildLinearBottleneck(
          bottleneck10, '11', {padding, groups: 576});
      const bottleneck12 = await buildLinearBottleneck(
          bottleneck11, '12', {padding, groups: 576});
      const bottleneck13 = await buildLinearBottleneck(
          bottleneck12, '13', {strides, padding, groups: 576}, false);
      const bottleneck14 = await buildLinearBottleneck(
          bottleneck13, '14', {padding, groups: 960});
      const bottleneck15 = await buildLinearBottleneck(
          bottleneck14, '15', {padding, groups: 960});
      const bottleneck16 = await buildLinearBottleneck(
          bottleneck15, '16', {padding, groups: 960}, false);
      const batch1 = await buildConvBatchNorm(bottleneck16, '1', '', true);
      const pool0 = builder.averagePool2d(batch1);
      const conv0WeightUrl =
          `${testDataDir}/weights/mobilenetv20_output_pred_weight.npy`;
      const conv0Weight = await utils.buildConstantFromNpy(
          builder, new URL(conv0WeightUrl, url));
      const conv0 = builder.conv2d(pool0, conv0Weight);
      const reshape0 = builder.reshape(conv0, [1, null]);
      const mobileNetGraph = await builder.build({reshape0});
      return mobileNetGraph;
    }
    graph = await buildMobileNet();
    fusedBatchNorm = true;
    fusedGraph = await buildMobileNet();
  });

  after(() => {
    if (typeof _tfengine !== 'undefined') {
      // Check memory leaks.
      graph.dispose();
      fusedGraph.dispose();
      const afterNumTensors = _tfengine.memory().numTensors;
      const afterNumBytes = _tfengine.memory().numBytes;
      assert(
          beforeNumTensors === afterNumTensors,
          `${afterNumTensors - beforeNumTensors} tensors are leaked.`);
      assert(
          beforeNumBytes === afterNumBytes,
          `${afterNumBytes - beforeNumBytes} bytes are leaked.`);
    }
  });

  async function testMobileNetV2(graph, inputFile, expectedFile) {
    const inputs = {
      'input': await utils.createTypedArrayFromNpy(new URL(inputFile, url)),
    };
    const outputs = {
      'reshape0': new Float32Array(utils.sizeOfShape([1, 1000])),
    };
    await context.compute(graph, inputs, outputs);
    const expected =
        await utils.createTypedArrayFromNpy(new URL(expectedFile, url));
    utils.checkValue(
        outputs.reshape0, expected, utils.modelFp32AccuracyCriteria);
  }

  it('test_data_set_0', async () => {
    await testMobileNetV2(
        graph, `${testDataDir}/test_data_set/0/input_0.npy`,
        `${testDataDir}/test_data_set/0/output_0.npy`);
  });

  it('test_data_set_1', async () => {
    await testMobileNetV2(
        graph, `${testDataDir}/test_data_set/1/input_0.npy`,
        `${testDataDir}/test_data_set/1/output_0.npy`);
  });

  it('test_data_set_2', async () => {
    await testMobileNetV2(
        graph, `${testDataDir}/test_data_set/2/input_0.npy`,
        `${testDataDir}/test_data_set/2/output_0.npy`);
  });

  it('test_data_set_0 (fused ops)', async () => {
    await testMobileNetV2(
        fusedGraph, `${testDataDir}/test_data_set/0/input_0.npy`,
        `${testDataDir}/test_data_set/0/output_0.npy`);
  });

  it('test_data_set_1 (fused ops)', async () => {
    await testMobileNetV2(
        fusedGraph, `${testDataDir}/test_data_set/1/input_0.npy`,
        `${testDataDir}/test_data_set/1/output_0.npy`);
  });

  it('test_data_set_2 (fused ops)', async () => {
    await testMobileNetV2(
        fusedGraph, `${testDataDir}/test_data_set/2/input_0.npy`,
        `${testDataDir}/test_data_set/2/output_0.npy`);
  });
});
