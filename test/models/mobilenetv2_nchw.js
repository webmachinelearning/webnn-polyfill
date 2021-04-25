'use strict';
import * as utils from '../utils.js';

const url = import.meta.url;
const assert = chai.assert;
const testDataDir = '../../test-data/models/mobilenetv2_nchw';

describe('test mobilenetv2 nchw', function() {
  // eslint-disable-next-line no-invalid-this
  this.timeout(0);
  let graph;
  let beforeNumBytes;
  let beforeNumTensors;
  before(async () => {
    if (typeof _tfengine !== 'undefined') {
      beforeNumBytes = _tfengine.memory().numBytes;
      beforeNumTensors = _tfengine.memory().numTensors;
    }
    const context = navigator.ml.createContext();
    const builder = new MLGraphBuilder(context);

    async function buildConv(input, name, relu6 = true, options = undefined) {
      const prefix = testDataDir + '/weights/conv_' + name;
      const weightsName = prefix + '_weight.npy';
      const weights =
          await utils.buildConstantFromNpy(builder, new URL(weightsName, url));
      const biasName = prefix + '_bias.npy';
      const bias =
          await utils.buildConstantFromNpy(builder, new URL(biasName, url));
      const conv = builder.add(
          builder.conv2d(input, weights, options),
          builder.reshape(bias, [1, -1, 1, 1]));
      if (relu6) {
        return builder.clamp(
          conv,
          {
            minValue: builder.constant(0.),
            maxValue: builder.constant(6.0),
          },
        );
      }
      return conv;
    }

    async function buildGemm(input, name) {
      const prefix = testDataDir + '/weights/gemm_' + name;
      const weightsName = prefix + '_weight.npy';
      const weights =
          await utils.buildConstantFromNpy(builder, new URL(weightsName, url));
      const biasName = prefix + '_bias.npy';
      const bias =
          await utils.buildConstantFromNpy(builder, new URL(biasName, url));
      const options = {c: bias, bTranspose: true};
      return builder.gemm(input, weights, options);
    }

    async function buildLinearBottleneck(
        input, convNameArray, group, stride, shortcut = true) {
      const conv1x1Relu6 = await buildConv(input, convNameArray[0]);
      const options = {
        padding: [1, 1, 1, 1],
        groups: group,
        strides: [stride, stride],
      };
      const dwise3x3Relu6 = await buildConv(
          conv1x1Relu6, convNameArray[1], true, options);
      const conv1x1Linear = await buildConv(
          dwise3x3Relu6, convNameArray[2], false);

      if (shortcut) {
        return builder.add(input, conv1x1Linear);
      }
      return conv1x1Linear;
    }

    const data =
        builder.input('input', {type: 'float32', dimensions: [1, 3, 224, 224]});
    const conv0 = await buildConv(
        data, '0', true, {padding: [1, 1, 1, 1], strides: [2, 2]});
    const conv1 = await buildConv(
        conv0, '2', true, {padding: [1, 1, 1, 1], groups: 32});
    const conv2 = await buildConv(conv1, '4', false);
    const bottleneck0 = await buildLinearBottleneck(
        conv2, ['5', '7', '9'], 96, 2, false);
    const bottleneck1 = await buildLinearBottleneck(
        bottleneck0, ['10', '12', '14'], 144, 1);
    const bottleneck2 = await buildLinearBottleneck(
        bottleneck1, ['16', '18', '20'], 144, 2, false);
    const bottleneck3 = await buildLinearBottleneck(
        bottleneck2, ['21', '23', '25'], 192, 1);
    const bottleneck4 = await buildLinearBottleneck(
        bottleneck3, ['27', '29', '31'], 192, 1);
    const bottleneck5 = await buildLinearBottleneck(
        bottleneck4, ['33', '35', '37'], 192, 2, false);
    const bottleneck6 = await buildLinearBottleneck(
        bottleneck5, ['38', '40', '42'], 384, 1);
    const bottleneck7 = await buildLinearBottleneck(
        bottleneck6, ['44', '46', '48'], 384, 1);
    const bottleneck8 = await buildLinearBottleneck(
        bottleneck7, ['50', '52', '54'], 384, 1);
    const bottleneck9 = await buildLinearBottleneck(
        bottleneck8, ['56', '58', '60'], 384, 1, false);
    const bottleneck10 = await buildLinearBottleneck(
        bottleneck9, ['61', '63', '65'], 576, 1);
    const bottleneck11 = await buildLinearBottleneck(
        bottleneck10, ['67', '69', '71'], 576, 1);
    const bottleneck12 = await buildLinearBottleneck(
        bottleneck11, ['73', '75', '77'], 576, 2, false);
    const bottleneck13 = await buildLinearBottleneck(
        bottleneck12, ['78', '80', '82'], 960, 1);
    const bottleneck14 = await buildLinearBottleneck(
        bottleneck13, ['84', '86', '88'], 960, 1);
    const bottleneck15 = await buildLinearBottleneck(
        bottleneck14, ['90', '92', '94'], 960, 1, false);

    const conv3 = await buildConv(bottleneck15, '95', true);
    const pool = builder.averagePool2d(conv3);
    const reshape = builder.reshape(pool, [1, -1]);
    const gemm = await buildGemm(reshape, '104');
    graph = await builder.build({gemm});
  });

  after(async () => {
    if (typeof _tfengine !== 'undefined') {
      // Check memory leaks.
      graph.dispose();
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

  async function testMobileNetV2(inputFile, expectedFile) {
    const input = await utils.createTypedArrayFromNpy(new URL(inputFile, url));
    const expected =
        await utils.createTypedArrayFromNpy(new URL(expectedFile, url));
    const outputs = await graph.compute({'input': {data: input}});
    utils.checkShape(outputs.gemm.dimensions, [1, 1000]);
    utils.checkValue(
        outputs.gemm.data, expected,
        new utils.AccuracyCriterion(1e-5, 1e-3));
  }

  it('test_data_set_0', async function() {
    await testMobileNetV2(
        `${testDataDir}/test_data_set/0/input_0.npy`,
        `${testDataDir}/test_data_set/0/output_0.npy`);
  });

  it('test_data_set_1', async function() {
    await testMobileNetV2(
        `${testDataDir}/test_data_set/1/input_0.npy`,
        `${testDataDir}/test_data_set/1/output_0.npy`);
  });

  it('test_data_set_2', async function() {
    await testMobileNetV2(
        `${testDataDir}/test_data_set/2/input_0.npy`,
        `${testDataDir}/test_data_set/2/output_0.npy`);
  });
});
