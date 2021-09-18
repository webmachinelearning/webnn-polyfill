'use strict';
import * as utils from '../utils.js';

const url = import.meta.url;
const assert = chai.assert;
const testDataDir = '../../test-data/models/tiny_yolov2_nhwc';

describe('test tinyYolov2 nhwc', function() {
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
    let fused = false;

    async function buildConv(input, name, leakyRelu = true) {
      const prefix = testDataDir + '/weights/conv2d_' + name;
      const weightsName = prefix + '_kernel.npy';
      const weights =
          await utils.buildConstantFromNpy(builder, new URL(weightsName, url));
      const biasName = prefix + '_Conv2D_bias.npy';
      const bias =
          await utils.buildConstantFromNpy(builder, new URL(biasName, url));
      const options = {
        inputLayout: 'nhwc',
        filterLayout: 'ohwi',
        autoPad: 'same-upper',
      };
      if (!fused) {
        let conv = builder.add(
            builder.conv2d(input, weights, options),
            builder.reshape(bias, [1, 1, 1, -1]));
        if (leakyRelu) {
          conv = builder.leakyRelu(conv, {alpha: 0.10000000149011612});
        }
        return conv;
      } else {
        options.bias = bias;
        if (leakyRelu) {
          options.activation = builder.leakyRelu({alpha: 0.10000000149011612});
        }
        return builder.conv2d(input, weights, options);
      }
    }

    async function buildTinyYolo() {
      const poolOptions = {
        windowDimensions: [2, 2],
        strides: [2, 2],
        autoPad: 'same-upper',
        layout: 'nhwc',
      };
      const data = builder.input(
          'input', {type: 'float32', dimensions: [1, 416, 416, 3]});
      const conv1 = await buildConv(data, '1');
      const pool1 = builder.maxPool2d(conv1, poolOptions);
      const conv2 = await buildConv(pool1, '2');
      const pool2 = builder.maxPool2d(conv2, poolOptions);
      const conv3 = await buildConv(pool2, '3');
      const pool3 = builder.maxPool2d(conv3, poolOptions);
      const conv4 = await buildConv(pool3, '4');
      const pool4 = builder.maxPool2d(conv4, poolOptions);
      const conv5 = await buildConv(pool4, '5');
      const pool5 = builder.maxPool2d(conv5, poolOptions);
      const conv6 = await buildConv(pool5, '6');
      const pool6 = builder.maxPool2d(
          conv6,
          {windowDimensions: [2, 2], autoPad: 'same-upper', layout: 'nhwc'});
      const conv7 = await buildConv(pool6, '7');
      const conv8 = await buildConv(conv7, '8');
      const conv = await buildConv(conv8, '9', false);
      return builder.build({conv});
    }

    graph = await buildTinyYolo();
    fused = true;
    fusedGraph = await buildTinyYolo();
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

  async function testTinyYoloV2(graph, inputFile, expectedFile) {
    const inputs = {
      'input': await utils.createTypedArrayFromNpy(new URL(inputFile, url)),
    };
    const outputs = {
      'conv': new Float32Array(utils.sizeOfShape([1, 13, 13, 125])),
    };
    graph.compute(inputs, outputs);
    const expected =
        await utils.createTypedArrayFromNpy(new URL(expectedFile, url));
    utils.checkValue(
        outputs.conv, expected, utils.ctsFp32RelaxedAccuracyCriteria);
  }

  it('test_data_set_0', async function() {
    await testTinyYoloV2(
        graph, `${testDataDir}/test_data_set/0/input_0.npy`,
        `${testDataDir}/test_data_set/0/output_0.npy`);
  });

  it('test_data_set_1', async function() {
    await testTinyYoloV2(
        graph, `${testDataDir}/test_data_set/1/input_0.npy`,
        `${testDataDir}/test_data_set/1/output_0.npy`);
  });

  it('test_data_set_2', async function() {
    await testTinyYoloV2(
        graph, `${testDataDir}/test_data_set/2/input_0.npy`,
        `${testDataDir}/test_data_set/2/output_0.npy`);
  });

  it('test_data_set_0 (fused ops)', async function() {
    await testTinyYoloV2(
        fusedGraph, `${testDataDir}/test_data_set/0/input_0.npy`,
        `${testDataDir}/test_data_set/0/output_0.npy`);
  });

  it('test_data_set_1 (fused ops)', async function() {
    await testTinyYoloV2(
        fusedGraph, `${testDataDir}/test_data_set/1/input_0.npy`,
        `${testDataDir}/test_data_set/1/output_0.npy`);
  });

  it('test_data_set_2 (fused ops)', async function() {
    await testTinyYoloV2(
        fusedGraph, `${testDataDir}/test_data_set/2/input_0.npy`,
        `${testDataDir}/test_data_set/2/output_0.npy`);
  });
});
