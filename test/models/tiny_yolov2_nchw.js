'use strict';
import * as utils from '../utils.js';

const url = import.meta.url;
const assert = chai.assert;
const testDataDir = '../../test-data/models/tiny_yolov2_nchw';

describe('test tinyYolov2 nchw', function() {
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
    let fused = false;

    async function buildConv(input, name, useBias = false) {
      const prefix = testDataDir + '/weights/convolution' + name;
      const weightName = prefix + '_W.npy';
      const weight =
          await utils.buildConstantFromNpy(builder, new URL(weightName, url));
      const options = {autoPad: 'same-upper'};
      options.padding =
          utils.computePadding2DForAutoPad(
              /* nchw */[input.shape()[2], input.shape()[3]],
              /* oihw */[weight.shape()[2], weight.shape()[3]],
              options.strides, options.dilations, options.autoPad);
      let bias;
      if (useBias) {
        const biasName = prefix + '_B.npy';
        bias =
            await utils.buildConstantFromNpy(builder, new URL(biasName, url));
        if (fused) {
          options.bias = bias;
        }
      }
      let conv = builder.conv2d(input, weight, options);
      if (useBias && !fused) {
        conv = builder.add(conv,
            builder.reshape(bias, [1, bias.shape()[0], 1, 1]));
      }
      return conv;
    }

    async function buildBatchNorm(input, name) {
      const prefix = testDataDir + '/weights/BatchNormalization';
      const scaleName = `${prefix}_scale${name}.npy`;
      const biasName = `${prefix}_B${name}.npy`;
      const meanName = `${prefix}_mean${name}.npy`;
      const varName = `${prefix}_variance${name}.npy`;
      const scale =
          await utils.buildConstantFromNpy(builder, new URL(scaleName, url));
      const bias =
          await utils.buildConstantFromNpy(builder, new URL(biasName, url));
      const mean =
          await utils.buildConstantFromNpy(builder, new URL(meanName, url));
      const variance =
          await utils.buildConstantFromNpy(builder, new URL(varName, url));

      if (!fused) {
        const batchNorm = builder.batchNormalization(
            input, mean, variance, {scale: scale, bias: bias});
        return builder.leakyRelu(batchNorm, {alpha: 0.10000000149011612});
      } else {
        return builder.batchNormalization(input, mean, variance, {
          scale: scale,
          bias: bias,
          activation: builder.leakyRelu({alpha: 0.10000000149011612}),
        });
      }
    }

    async function buildConvolutional(input, name) {
      const conv = await buildConv(input, name);
      return await buildBatchNorm(conv, name);
    }

    async function buildTinyYolo() {
      const mulScale = builder.constant(
          {dataType: 'float32', dimensions: [1]},
          new Float32Array([0.003921568859368563]));
      const addBias = builder.constant(
          {dataType: 'float32', dimensions: [3, 1, 1]},
          new Float32Array([0, 0, 0]));
      const poolOptions = {
        windowDimensions: [2, 2],
        strides: [2, 2],
        autoPad: 'same-upper',
      };
      const data = builder.input(
          'input', {dataType: 'float32', dimensions: [1, 3, 416, 416]});
      const mul = builder.mul(data, mulScale);
      const add = builder.add(mul, addBias);
      const conv0 = await buildConvolutional(add, '');
      const pool0 = utils.buildMaxPool2d(conv0, poolOptions, builder, 'nchw');
      const conv1 = await buildConvolutional(pool0, '1');
      const pool1 = utils.buildMaxPool2d(conv1, poolOptions, builder, 'nchw');
      const conv2 = await buildConvolutional(pool1, '2');
      const pool2 = utils.buildMaxPool2d(conv2, poolOptions, builder, 'nchw');
      const conv3 = await buildConvolutional(pool2, '3');
      const pool3 = utils.buildMaxPool2d(conv3, poolOptions, builder, 'nchw');
      const conv4 = await buildConvolutional(pool3, '4');
      const pool4 = utils.buildMaxPool2d(conv4, poolOptions, builder, 'nchw');
      const conv5 = await buildConvolutional(pool4, '5');
      const pool5 = utils.buildMaxPool2d(
          conv5, {windowDimensions: [2, 2], autoPad: 'same-upper'}, builder,
          'nchw');
      const conv6 = await buildConvolutional(pool5, '6');
      const conv7 = await buildConvolutional(conv6, '7');
      const conv = await buildConv(conv7, '8', true);
      const tinyYoloGraph = await builder.build({conv});
      return tinyYoloGraph;
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
      'conv': new Float32Array(utils.sizeOfShape([1, 125, 13, 13])),
    };
    const result = await context.compute(graph, inputs, outputs);
    const expected =
        await utils.createTypedArrayFromNpy(new URL(expectedFile, url));
    utils.checkValue(
        result.outputs.conv, expected,
        // refer to onnx
        // https://github.com/onnx/models/blob/master/workflow_scripts/ort_test_dir_utils.py#L239
        new utils.AccuracyCriterion(1e-3, 1e-3));
  }

  it('test_data_set_0', async () => {
    await testTinyYoloV2(
        graph, `${testDataDir}/test_data_set/0/input_0.npy`,
        `${testDataDir}/test_data_set/0/output_0.npy`);
  });

  it('test_data_set_1', async () => {
    await testTinyYoloV2(
        graph, `${testDataDir}/test_data_set/1/input_0.npy`,
        `${testDataDir}/test_data_set/1/output_0.npy`);
  });

  it('test_data_set_2', async () => {
    await testTinyYoloV2(
        graph, `${testDataDir}/test_data_set/2/input_0.npy`,
        `${testDataDir}/test_data_set/2/output_0.npy`);
  });

  it('test_data_set_0 (fused ops)', async () => {
    await testTinyYoloV2(
        fusedGraph, `${testDataDir}/test_data_set/0/input_0.npy`,
        `${testDataDir}/test_data_set/0/output_0.npy`);
  });

  it('test_data_set_1 (fused ops)', async () => {
    await testTinyYoloV2(
        fusedGraph, `${testDataDir}/test_data_set/1/input_0.npy`,
        `${testDataDir}/test_data_set/1/output_0.npy`);
  });

  it('test_data_set_2 (fused ops)', async () => {
    await testTinyYoloV2(
        fusedGraph, `${testDataDir}/test_data_set/2/input_0.npy`,
        `${testDataDir}/test_data_set/2/output_0.npy`);
  });
});
