'use strict';
import * as utils from '../utils.js';

const url = import.meta.url;
const assert = chai.assert;
const testDataDir = '../../test-data/models/squeezenet1.1_nchw';

describe('test squeezenet1.1 nchw', function() {
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
    let fusedConv = false;

    async function buildConv(input, name, options = {}) {
      const prefix = testDataDir + '/weights/squeezenet0_' + name;
      const weightsName = prefix + '_weight.npy';
      const weights =
          await utils.buildConstantFromNpy(builder, new URL(weightsName, url));
      const biasName = prefix + '_bias.npy';
      const bias =
          await utils.buildConstantFromNpy(builder, new URL(biasName, url));
      if (!fusedConv) {
        return builder.relu(builder.add(
            builder.conv2d(input, weights, options),
            builder.reshape(bias, [1, -1, 1, 1])));
      } else {
        options.bias = bias;
        options.activation = builder.relu();
        return builder.conv2d(input, weights, options);
      }
    }

    async function buildFire(input, convName, conv1x1Name, conv3x3Name) {
      const conv = await buildConv(input, convName);
      const conv1x1 = await buildConv(conv, conv1x1Name);
      const conv3x3 =
          await buildConv(conv, conv3x3Name, {padding: [1, 1, 1, 1]});
      return builder.concat([conv1x1, conv3x3], 1);
    }

    async function buildSqueezeNet() {
      const data = builder.input(
          'data', {type: 'float32', dimensions: [1, 3, 224, 224]});
      const conv0 = await buildConv(data, 'conv0', {strides: [2, 2]});
      const pool0 =
          builder.maxPool2d(conv0, {windowDimensions: [3, 3], strides: [2, 2]});
      const fire0 = await buildFire(pool0, 'conv1', 'conv2', 'conv3');
      const fire1 = await buildFire(fire0, 'conv4', 'conv5', 'conv6');
      const pool1 =
          builder.maxPool2d(fire1, {windowDimensions: [3, 3], strides: [2, 2]});
      const fire2 = await buildFire(pool1, 'conv7', 'conv8', 'conv9');
      const fire3 = await buildFire(fire2, 'conv10', 'conv11', 'conv12');
      const pool2 =
          builder.maxPool2d(fire3, {windowDimensions: [3, 3], strides: [2, 2]});
      const fire4 = await buildFire(pool2, 'conv13', 'conv14', 'conv15');
      const fire5 = await buildFire(fire4, 'conv16', 'conv17', 'conv18');
      const fire6 = await buildFire(fire5, 'conv19', 'conv20', 'conv21');
      const fire7 = await buildFire(fire6, 'conv22', 'conv23', 'conv24');
      const conv25 = await buildConv(fire7, 'conv25');
      const pool3 = builder.averagePool2d(
          conv25, {windowDimensions: [13, 13], strides: [13, 13]});
      const reshape0 = builder.reshape(pool3, [1, -1]);
      return builder.build({reshape0});
    }

    graph = await buildSqueezeNet();
    fusedConv = true;
    fusedGraph = await buildSqueezeNet();
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

  async function testSqueezeNet(graph, inputFile, expectedFile) {
    const inputs = {
      'data': await utils.createTypedArrayFromNpy(new URL(inputFile, url)),
    };
    const outputs = {
      'reshape0': new Float32Array(utils.sizeOfShape([1, 1000])),
    };
    graph.compute(inputs, outputs);
    const expected =
        await utils.createTypedArrayFromNpy(new URL(expectedFile, url));
    utils.checkValue(
        outputs.reshape0, expected, utils.modelFp32AccuracyCriteria);
  }

  it('test_data_set_0', async function() {
    await testSqueezeNet(
        graph, `${testDataDir}/test_data_set/0/input_0.npy`,
        `${testDataDir}/test_data_set/0/output_0.npy`);
  });

  it('test_data_set_1', async function() {
    await testSqueezeNet(
        graph, `${testDataDir}/test_data_set/1/input_0.npy`,
        `${testDataDir}/test_data_set/1/output_0.npy`);
  });

  it('test_data_set_2', async function() {
    await testSqueezeNet(
        graph, `${testDataDir}/test_data_set/2/input_0.npy`,
        `${testDataDir}/test_data_set/2/output_0.npy`);
  });

  it('test_data_set_0 (fused ops)', async function() {
    await testSqueezeNet(
        fusedGraph, `${testDataDir}/test_data_set/0/input_0.npy`,
        `${testDataDir}/test_data_set/0/output_0.npy`);
  });

  it('test_data_set_1 (fused ops)', async function() {
    await testSqueezeNet(
        fusedGraph, `${testDataDir}/test_data_set/1/input_0.npy`,
        `${testDataDir}/test_data_set/1/output_0.npy`);
  });

  it('test_data_set_2 (fused ops)', async function() {
    await testSqueezeNet(
        fusedGraph, `${testDataDir}/test_data_set/2/input_0.npy`,
        `${testDataDir}/test_data_set/2/output_0.npy`);
  });
});
