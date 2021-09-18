'use strict';
import * as utils from '../utils.js';

/* eslint max-len: ["error", {"code": 120}] */

const url = import.meta.url;
const assert = chai.assert;
const testDataDir = '../../test-data/models/resnet50v2_nchw';

describe('test resnet50v2 nchw', function() {
  // eslint-disable-next-line no-invalid-this
  this.timeout(0);
  let graph;
  let fusedGraph;
  let beforeNumBytes;
  let beforeNumTensors;
  before(async () => {
    if (typeof _tfengine !== 'undefined') {
      beforeNumBytes = _tfengine.memory().numBytes;
      beforeNumTensors = _tfengine.memory().numTensors;
    }
    const context = navigator.ml.createContext();
    const builder = new MLGraphBuilder(context);
    let fusedBatchNorm = false;

    async function buildConv(input, name, stageName, options = undefined) {
      let prefix = '';
      if (stageName !== '') {
        prefix = `${testDataDir}/weights/resnetv24_stage${stageName}_conv` +
              name;
      } else {
        prefix = `${testDataDir}/weights/resnetv24_conv${name}`;
      }
      const weightName = prefix + '_weight.npy';
      const weight =
            await utils.buildConstantFromNpy(builder, new URL(weightName, url));
      return builder.conv2d(input, weight, options);
    }

    async function buildBatchNorm(input, name, stageName, relu = true) {
      let prefix = '';
      if (stageName !== '') {
        prefix = `${testDataDir}/weights/resnetv24_stage${stageName}` +
            '_batchnorm' + name;
      } else {
        prefix = `${testDataDir}/weights/resnetv24_batchnorm${name}`;
      }
      const scaleName = prefix + '_gamma.npy';
      const biasName = prefix + '_beta.npy';
      const meanName = prefix + '_running_mean.npy';
      const varName = prefix + '_running_var.npy';
      const scale =
          await utils.buildConstantFromNpy(builder, new URL(scaleName, url));
      const bias =
          await utils.buildConstantFromNpy(builder, new URL(biasName, url));
      const mean =
          await utils.buildConstantFromNpy(builder, new URL(meanName, url));
      const variance =
          await utils.buildConstantFromNpy(builder, new URL(varName, url));
      const options = {scale: scale, bias: bias};
      if (!fusedBatchNorm) {
        const batchNorm =
            builder.batchNormalization(input, mean, variance, options);
        if (relu) {
          return builder.relu(batchNorm);
        }
        return batchNorm;
      } else {
        if (relu) {
          options.activation = utils.createActivation(builder, 'relu');
        } else {
          options.activation = undefined;
        }
        return builder.batchNormalization(input, mean, variance, options);
      }
    }

    async function buildGemm(input, name) {
      const prefix = `${testDataDir}/weights/resnetv24_dense${name}`;
      const weightName = prefix + '_weight.npy';
      const weight =
        await utils.buildConstantFromNpy(builder, new URL(weightName, url));
      const biasName = prefix + '_bias.npy';
      const bias =
        await utils.buildConstantFromNpy(builder, new URL(biasName, url));
      const options = {c: bias, bTranspose: true};
      return builder.gemm(input, weight, options);
    }

    async function buildBottlenectV2(
        input, stageName, nameIndices, downsample = false, stride = 1) {
      let residual = input;
      let strides = [1, 1];

      if (downsample) {
        strides = [stride, stride];
      }
      const bn1 = await buildBatchNorm(input, nameIndices[0], stageName);
      const conv1 = await buildConv(bn1, nameIndices[1], stageName);
      const bn2 = await buildBatchNorm(
          conv1, parseInt(nameIndices[0]) + 1, stageName);
      const conv2 = await buildConv(
          bn2, nameIndices[2], stageName, {padding: [1, 1, 1, 1], strides});
      const bn3 = await buildBatchNorm(
          conv2, parseInt(nameIndices[0]) + 2, stageName);
      const conv3 = await buildConv(bn3, nameIndices[3], stageName);
      if (downsample) {
        residual = await buildConv(
            bn1, parseInt(nameIndices[0]) + 3, stageName, {strides});
      }
      return builder.add(conv3, residual);
    }

    async function buildResNet() {
      const data =
          builder.input('input', {type: 'float32', dimensions: [1, 3, 224, 224]});
      const bn1 = await buildBatchNorm(data, '0', '', false);
      const conv0 = await buildConv(
          bn1, '0', '', {padding: [3, 3, 3, 3], strides: [2, 2]});
      const bn2 = await buildBatchNorm(conv0, '1', '');
      const pool1 = await builder.maxPool2d(bn2,
          {windowDimensions: [3, 3], padding: [1, 1, 1, 1], strides: [2, 2]});

      // Stage 1
      const bottleneck1 = await buildBottlenectV2(
          pool1, '1', ['0', '0', '1', '2'], true);
      const bottleneck2 = await buildBottlenectV2(
          bottleneck1, '1', ['3', '4', '5', '6']);
      const bottleneck3 = await buildBottlenectV2(
          bottleneck2, '1', ['6', '7', '8', '9']);

      // Stage 2
      const bottleneck4 = await buildBottlenectV2(
          bottleneck3, '2', ['0', '0', '1', '2'], true, 2);
      const bottleneck5 = await buildBottlenectV2(
          bottleneck4, '2', ['3', '4', '5', '6']);
      const bottleneck6 = await buildBottlenectV2(
          bottleneck5, '2', ['6', '7', '8', '9']);
      const bottleneck7 = await buildBottlenectV2(
          bottleneck6, '2', ['9', '10', '11', '12']);

      // Stage 3
      const bottleneck8 = await buildBottlenectV2(
          bottleneck7, '3', ['0', '0', '1', '2'], true, 2);
      const bottleneck9 = await buildBottlenectV2(
          bottleneck8, '3', ['3', '4', '5', '6']);
      const bottleneck10 = await buildBottlenectV2(
          bottleneck9, '3', ['6', '7', '8', '9']);
      const bottleneck11 = await buildBottlenectV2(
          bottleneck10, '3', ['9', '10', '11', '12']);
      const bottleneck12 = await buildBottlenectV2(
          bottleneck11, '3', ['12', '13', '14', '15']);
      const bottleneck13 = await buildBottlenectV2(
          bottleneck12, '3', ['15', '16', '17', '18']);

      // Stage 4
      const bottleneck14 = await buildBottlenectV2(
          bottleneck13, '4', ['0', '0', '1', '2'], true, 2);
      const bottleneck15 = await buildBottlenectV2(
          bottleneck14, '4', ['3', '4', '5', '6']);
      const bottleneck16 = await buildBottlenectV2(
          bottleneck15, '4', ['6', '7', '8', '9']);

      const bn3 = await buildBatchNorm(bottleneck16, '2', '');
      const pool2 = await builder.averagePool2d(bn3);
      const reshape = builder.reshape(pool2, [1, -1]);
      const gemm = await buildGemm(reshape, '0');
      return builder.build({gemm});
    }

    graph = await buildResNet();
    fusedBatchNorm = true;
    fusedGraph = await buildResNet();
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

  async function testResNet50V2(graph, inputFile, expectedFile) {
    const inputs = {
      'input': await utils.createTypedArrayFromNpy(new URL(inputFile, url))};
    const outputs = {
      'gemm': new Float32Array(utils.sizeOfShape([1, 1000]))};
    graph.compute(inputs, outputs);
    const expected =
        await utils.createTypedArrayFromNpy(new URL(expectedFile, url));
    utils.checkValue(
        outputs.gemm, expected,
        // refer to onnx
        // https://github.com/onnx/models/blob/master/workflow_scripts/ort_test_dir_utils.py#L239
        new utils.AccuracyCriterion(1e-3, 1e-3));
  }

  it('test_data_set_0', async function() {
    await testResNet50V2(
        graph, `${testDataDir}/test_data_set/0/input_0.npy`,
        `${testDataDir}/test_data_set/0/output_0.npy`);
  });

  it('test_data_set_1', async function() {
    await testResNet50V2(
        graph, `${testDataDir}/test_data_set/1/input_0.npy`,
        `${testDataDir}/test_data_set/1/output_0.npy`);
  });

  it('test_data_set_2', async function() {
    await testResNet50V2(
        graph, `${testDataDir}/test_data_set/2/input_0.npy`,
        `${testDataDir}/test_data_set/2/output_0.npy`);
  });

  it('test_data_set_0 (fused ops)', async function() {
    await testResNet50V2(
        fusedGraph, `${testDataDir}/test_data_set/0/input_0.npy`,
        `${testDataDir}/test_data_set/0/output_0.npy`);
  });

  it('test_data_set_1 (fused ops)', async function() {
    await testResNet50V2(
        fusedGraph, `${testDataDir}/test_data_set/1/input_0.npy`,
        `${testDataDir}/test_data_set/1/output_0.npy`);
  });

  it('test_data_set_2 (fused ops)', async function() {
    await testResNet50V2(
        fusedGraph, `${testDataDir}/test_data_set/2/input_0.npy`,
        `${testDataDir}/test_data_set/2/output_0.npy`);
  });
});
