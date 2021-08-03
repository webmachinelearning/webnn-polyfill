'use strict';
import * as utils from '../utils.js';

const url = import.meta.url;
const assert = chai.assert;
const testDataDir = '../../test-data/models/squeezenet1.0_nhwc';

describe('test squeezenet1.0 nhwc', function() {
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
      const prefix = testDataDir + '/weights/' + name;
      const weightsName = prefix + '_kernel.npy';
      const weights =
          await utils.buildConstantFromNpy(builder, new URL(weightsName, url));
      const biasName = prefix + '_Conv2D_bias.npy';
      const bias =
          await utils.buildConstantFromNpy(builder, new URL(biasName, url));
      options.inputLayout = 'nhwc';
      options.filterLayout = 'ohwi';

      if (fusedConv === false) {
        return builder.relu(builder.add(
            builder.conv2d(input, weights, options),
            builder.reshape(bias, [1, 1, 1, -1])));
      } else {
        options.bias = bias;
        options.activation = builder.relu();
        return builder.conv2d(input, weights, options);
      }
    }

    async function buildFire(input, name) {
      const convSqueeze = await buildConv(input, name + '_squeeze');
      const convE1x1 = await buildConv(convSqueeze, name + '_e1x1');
      const convE3x3 =
          await buildConv(convSqueeze, name + '_e3x3', {padding: [1, 1, 1, 1]});
      return builder.concat([convE1x1, convE3x3], 3);
    }

    async function buildSqueezeNet() {
      const strides = [2, 2];
      const layout = 'nhwc';
      const placeholder = builder.input(
          'placeholder', {type: 'float32', dimensions: [1, 224, 224, 3]});
      const conv1 = await buildConv(
          placeholder, 'conv1', {strides, autoPad: 'same-upper'});
      const maxpool1 =
          builder.maxPool2d(conv1, {windowDimensions: [3, 3], strides, layout});
      const fire2 = await buildFire(maxpool1, 'fire2');
      const fire3 = await buildFire(fire2, 'fire3');
      const fire4 = await buildFire(fire3, 'fire4');
      const maxpool4 =
          builder.maxPool2d(fire4, {windowDimensions: [3, 3], strides, layout});
      const fire5 = await buildFire(maxpool4, 'fire5');
      const fire6 = await buildFire(fire5, 'fire6');
      const fire7 = await buildFire(fire6, 'fire7');
      const fire8 = await buildFire(fire7, 'fire8');
      const maxpool8 =
          builder.maxPool2d(fire8, {windowDimensions: [3, 3], strides, layout});
      const fire9 = await buildFire(maxpool8, 'fire9');
      const conv10 = await buildConv(fire9, 'conv10');
      const averagePool2d =
          builder.averagePool2d(conv10, {windowDimensions: [13, 13], layout});
      const reshape = builder.reshape(averagePool2d, [1, -1]);
      const softmax = builder.softmax(reshape);
      return builder.build({softmax});
    }
    graph = await buildSqueezeNet();
    fusedConv = true;
    fusedGraph = await buildSqueezeNet();
  });

  after(async () => {
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
      'placeholder':
          await utils.createTypedArrayFromNpy(new URL(inputFile, url)),
    };
    const outputs = {'softmax': new Float32Array(utils.sizeOfShape([1, 1001]))};
    graph.compute(inputs, outputs);
    const expected =
        await utils.createTypedArrayFromNpy(new URL(expectedFile, url));
    utils.checkValue(
        outputs.softmax, expected, utils.modelFp32AccuracyCriteria);
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
        `${testDataDir}/test_data_set/0/output_0.npy`, true);
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
