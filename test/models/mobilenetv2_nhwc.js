'use strict';
import * as utils from '../utils.js';

/* eslint max-len: ["error", {"code": 120}] */

const url = import.meta.url;
const assert = chai.assert;
const testDataDir = '../../test-data/models/mobilenetv2_nhwc';

describe('test mobilenetv2 nhwc', function() {
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

    async function buildConv(input, weightsSubName, biasSubName, relu6, options) {
      const weightsName = `${testDataDir}/weights/Const_${weightsSubName}.npy`;
      const weights =
          await utils.buildConstantFromNpy(builder, new URL(weightsName, url));
      const biasName = `${testDataDir}/weights/MobilenetV2_${biasSubName}_bias.npy`;
      const bias =
          await utils.buildConstantFromNpy(builder, new URL(biasName, url));
      options.inputLayout = 'nhwc';
      const add = builder.add(
          builder.conv2d(input, weights, options),
          builder.reshape(bias, [1, 1, 1, -1]));
      // `relu6` in TFLite equals to `clamp` in WebNN API
      if (relu6) {
        return builder.clamp(
            add,
            {
              minValue: builder.constant(0.),
              maxValue: builder.constant(6.0),
            });
      }
      return add;
    }

    async function buildLinearBottleneck(input, weightsNameArray, biasName, dwiseOptions, shortcut = true) {
      const autoPad = 'same-upper';
      const biasPrefix = 'expanded_conv_' + biasName;

      dwiseOptions.autoPad = autoPad;
      dwiseOptions.filterLayout = 'ihwo';
      const convOptions = {autoPad, filterLayout: 'ohwi'};

      const conv1x1Relu6 = await buildConv(input, weightsNameArray[0],
          `${biasPrefix}_expand_Conv2D`, true, convOptions);
      const dwise3x3Relu6 = await buildConv(conv1x1Relu6, weightsNameArray[1],
          `${biasPrefix}_depthwise_depthwise`, true, dwiseOptions);
      const conv1x1Linear = await buildConv(dwise3x3Relu6, weightsNameArray[2],
          `${biasPrefix}_project_Conv2D`, false, convOptions);

      if (shortcut) {
        return builder.add(input, conv1x1Linear);
      }
      return conv1x1Linear;
    }

    const strides = [2, 2];
    const autoPad = 'same-upper';
    const filterLayout = 'ohwi';
    const data = builder.input(
        'input', {type: 'float32', dimensions: [1, 224, 224, 3]});
    const conv0 = await buildConv(
        data, '90', 'Conv_Conv2D', true, {strides, autoPad, filterLayout});
    const conv1 = await buildConv(
        conv0, '238', 'expanded_conv_depthwise_depthwise', true, {autoPad, groups: 32, filterLayout: 'ihwo'});
    const conv2 = await buildConv(
        conv1, '167', 'expanded_conv_project_Conv2D', false, {autoPad, filterLayout});
    const bottleneck0 = await buildLinearBottleneck(
        conv2, ['165', '99', '73'], '1', {strides, groups: 96}, false);
    const bottleneck1 = await buildLinearBottleneck(
        bottleneck0, ['3', '119', '115'], '2', {groups: 144});
    const bottleneck2 = await buildLinearBottleneck(
        bottleneck1, ['255', '216', '157'], '3', {strides, groups: 144}, false);
    const bottleneck3 = await buildLinearBottleneck(
        bottleneck2, ['227', '221', '193'], '4', {groups: 192});
    const bottleneck4 = await buildLinearBottleneck(
        bottleneck3, ['243', '102', '215'], '5', {groups: 192});
    const bottleneck5 = await buildLinearBottleneck(
        bottleneck4, ['226', '163', '229'], '6', {strides, groups: 192}, false);
    const bottleneck6 = await buildLinearBottleneck(
        bottleneck5, ['104', '254', '143'], '7', {groups: 384});
    const bottleneck7 = await buildLinearBottleneck(
        bottleneck6, ['25', '142', '202'], '8', {groups: 384});
    const bottleneck8 = await buildLinearBottleneck(
        bottleneck7, ['225', '129', '98'], '9', {groups: 384});
    const bottleneck9 = await buildLinearBottleneck(
        bottleneck8, ['169', '2', '246'], '10', {groups: 384}, false);
    const bottleneck10 = await buildLinearBottleneck(
        bottleneck9, ['162', '87', '106'], '11', {groups: 576});
    const bottleneck11 = await buildLinearBottleneck(
        bottleneck10, ['52', '22', '40'], '12', {groups: 576});
    const bottleneck12 = await buildLinearBottleneck(
        bottleneck11, ['114', '65', '242'], '13', {strides, groups: 576}, false);
    const bottleneck13 = await buildLinearBottleneck(
        bottleneck12, ['203', '250', '92'], '14', {groups: 960});
    const bottleneck14 = await buildLinearBottleneck(
        bottleneck13, ['133', '130', '258'], '15', {groups: 960});
    const bottleneck15 = await buildLinearBottleneck(
        bottleneck14, ['60', '248', '100'], '16', {groups: 960}, false);
    const conv3 = await buildConv(
        bottleneck15, '71', 'Conv_1_Conv2D', true, {autoPad, filterLayout});

    const averagePool2d = builder.averagePool2d(
        conv3, {windowDimensions: [7, 7], layout: 'nhwc'});
    const conv4 = await buildConv(
        averagePool2d, '222', 'Logits_Conv2d_1c_1x1_Conv2D', false, {autoPad, filterLayout});
    const reshape = builder.reshape(conv4, [1, -1]);
    const softmax = builder.softmax(reshape);
    graph = await builder.build({softmax});
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
    utils.checkShape(outputs.softmax.dimensions, [1, 1001]);
    utils.checkValue(
        outputs.softmax.data, expected,
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
