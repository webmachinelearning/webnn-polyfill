'use strict';
import * as utils from '../utils.js';

/* eslint max-len: ["error", {"code": 120}] */

const url = import.meta.url;
const assert = chai.assert;
const testDataDir = '../../test-data/models/resnet101v2_nhwc';

describe('test resnet101v2 nhwc', function() {
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
    let fusedConv = false;
    const autoPad = 'same-upper';
    const strides = [2, 2];
    const layout = 'nhwc';

    async function buildConv(input, nameIndices, options = undefined, relu = true) {
      let prefix = `${testDataDir}/weights/resnet_v2_101_`;
      // Items in 'nameIndices' represent the indices of block, unit, conv
      // respectively, except two kinds of specific conv names:
      // 1. contains 'shortcut', e.g.
      // resnet_v2_101_block1_unit_1_bottleneck_v2_shortcut_weights.npy
      // 2. contains 'logits', e.g. resnet_v2_101_logits_weights.npy
      if (nameIndices[0] !== '' && nameIndices[1] !== '') {
        prefix += `block${nameIndices[0]}_unit_${nameIndices[1]}_bottleneck_v2_`;
      }
      if (nameIndices[2] === 'shortcut') {
        prefix += 'shortcut';
      } else if (nameIndices[2] === 'logits') {
        prefix += nameIndices[2];
      } else {
        prefix += 'conv' + nameIndices[2];
      }
      const weightsName = prefix + '_weights.npy';
      const weights =
          await utils.buildConstantFromNpy(builder, new URL(weightsName, url));
      const biasName = prefix + '_Conv2D_bias.npy';
      const bias =
          await utils.buildConstantFromNpy(builder, new URL(biasName, url));
      if (options !== undefined) {
        options.inputLayout = layout;
        options.filterLayout = 'ohwi';
      } else {
        options = {inputLayout: layout, filterLayout: 'ohwi'};
      }
      if (!fusedConv) {
        const add = builder.add(
            builder.conv2d(input, weights, options),
            builder.reshape(bias, [1, 1, 1, null]));
        if (relu) {
          return builder.relu(add);
        }
        return add;
      } else {
        options.bias = bias;
        if (relu) {
          options.activation = utils.createActivation(builder, 'relu');
        } else {
          options.activation = undefined;
        }
        return builder.conv2d(input, weights, options);
      }
    }

    async function buildFusedBatchNorm(input, nameIndices) {
      let prefix = `${testDataDir}/weights/resnet_v2_101_`;
      if (nameIndices[0] === 'postnorm') {
        prefix += 'postnorm';
      } else {
        prefix +=
            `block${nameIndices[0]}_unit_${nameIndices[1]}_bottleneck_v2_preact`;
      }
      const mulParamName = prefix + '_FusedBatchNorm_mul_0_param.npy';
      const mulParam =
          await utils.buildConstantFromNpy(builder, new URL(mulParamName, url));
      const addParamName = prefix + '_FusedBatchNorm_add_param.npy';
      const addParam =
          await utils.buildConstantFromNpy(builder, new URL(addParamName, url));
      return builder.relu(
          builder.add(builder.mul(input, mulParam), addParam));
    }

    async function buildBottleneckV2(
        input, nameIndices, downsample = false, shortcut = true) {
      let residual = input;

      const fusedBn = await buildFusedBatchNorm(input, nameIndices);
      const conv1 = await buildConv(
          fusedBn, nameIndices.concat(['1']), {autoPad});
      let conv2;
      if (downsample) {
        residual = await buildConv(
            fusedBn, nameIndices.concat(['shortcut']), {autoPad}, false);
      }
      if (!downsample && shortcut) {
        residual = builder.maxPool2d(
            input, {windowDimensions: [1, 1], strides, layout, autoPad});
        const padding = builder.constant(
            {type: 'int32', dimensions: [4, 2]},
            new Int32Array([0, 0, 1, 1, 1, 1, 0, 0]));
        const pad = builder.pad(conv1, padding);
        conv2 = await buildConv(pad, nameIndices.concat(['2']), {strides});
      } else {
        conv2 = await buildConv(
            conv1, nameIndices.concat(['2']), {autoPad});
      }
      const conv3 = await buildConv(
          conv2, nameIndices.concat(['3']), {autoPad}, false);
      return builder.add(conv3, residual);
    }

    async function buildResNet() {
      const padding = builder.constant(
          {type: 'int32', dimensions: [4, 2]},
          new Int32Array([0, 0, 3, 3, 3, 3, 0, 0]));

      const input = builder.input('input',
          {type: 'float32', dimensions: [1, 299, 299, 3]});
      const pad = builder.pad(input, padding);
      const conv1 = await buildConv(pad, ['', '', '1'], {strides}, false);
      const pool = builder.maxPool2d(
          conv1, {windowDimensions: [3, 3], strides, layout, autoPad});
      // Block 1
      const bottleneck1 = await buildBottleneckV2(pool, ['1', '1'], true);
      const bottleneck2 =
          await buildBottleneckV2(bottleneck1, ['1', '2'], false, false);
      const bottleneck3 = await buildBottleneckV2(bottleneck2, ['1', '3']);

      // Block 2
      const bottleneck4 = await buildBottleneckV2(bottleneck3, ['2', '1'], true);
      const bottleneck5 =
          await buildBottleneckV2(bottleneck4, ['2', '2'], false, false);
      const bottleneck6 =
          await buildBottleneckV2(bottleneck5, ['2', '3'], false, false);
      const bottleneck7 = await buildBottleneckV2(bottleneck6, ['2', '4']);

      // Block 3
      const bottleneck8 = await buildBottleneckV2(bottleneck7, ['3', '1'], true);
      const loop = async (node, num) => {
        if (num > 22) {
          return node;
        } else {
          const newNode = await buildBottleneckV2(
              node, ['3', num.toString()], false, false);
          num++;
          return loop(newNode, num);
        }
      };
      const bottleneck9 = await loop(bottleneck8, 2);
      const bottleneck10 =await buildBottleneckV2(bottleneck9, ['3', '23']);

      // Block 4
      const bottleneck11 =
          await buildBottleneckV2(bottleneck10, ['4', '1'], true);
      const bottleneck12 =
          await buildBottleneckV2(bottleneck11, ['4', '2'], false, false);
      const bottleneck13 =
          await buildBottleneckV2(bottleneck12, ['4', '3'], false, false);

      const fusedBn =
          await buildFusedBatchNorm(bottleneck13, ['postnorm']);
      const mean =
          builder.reduceMean(fusedBn, {keepDimensions: true, axes: [1, 2]});
      const conv2 =
          await buildConv(mean, ['', '', 'logits'], {autoPad}, false);
      const reshape = builder.reshape(conv2, [1, null]);
      const resNetGraph = await builder.build({reshape});
      return resNetGraph;
    }

    graph = await buildResNet();
    fusedConv = true;
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

  async function testResNet101V2(graph, inputFile, expectedFile) {
    const inputs = {
      'input': await utils.createTypedArrayFromNpy(new URL(inputFile, url))};
    const outputs = {
      'reshape': new Float32Array(utils.sizeOfShape([1, 1001]))};
    await context.compute(graph, inputs, outputs);
    const expected =
        await utils.createTypedArrayFromNpy(new URL(expectedFile, url));
    utils.checkValue(
        outputs.reshape, expected, utils.ctsFp32RestrictAccuracyCriteria);
  }

  it('test_data_set_0', async () => {
    await testResNet101V2(
        graph, `${testDataDir}/test_data_set/0/input_0.npy`,
        `${testDataDir}/test_data_set/0/output_0.npy`);
  });

  it('test_data_set_1', async () => {
    await testResNet101V2(
        graph, `${testDataDir}/test_data_set/1/input_0.npy`,
        `${testDataDir}/test_data_set/1/output_0.npy`);
  });

  it('test_data_set_2', async () => {
    await testResNet101V2(
        graph, `${testDataDir}/test_data_set/2/input_0.npy`,
        `${testDataDir}/test_data_set/2/output_0.npy`);
  });

  it('test_data_set_0 (fused ops)', async () => {
    await testResNet101V2(
        fusedGraph, `${testDataDir}/test_data_set/0/input_0.npy`,
        `${testDataDir}/test_data_set/0/output_0.npy`);
  });

  it('test_data_set_1 (fused ops)', async () => {
    await testResNet101V2(
        fusedGraph, `${testDataDir}/test_data_set/1/input_0.npy`,
        `${testDataDir}/test_data_set/1/output_0.npy`);
  });

  it('test_data_set_2 (fused ops)', async () => {
    await testResNet101V2(
        fusedGraph, `${testDataDir}/test_data_set/2/input_0.npy`,
        `${testDataDir}/test_data_set/2/output_0.npy`);
  });
});
