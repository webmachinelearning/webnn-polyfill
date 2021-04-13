'use strict';
import * as utils from '../../utils.js';

const url = import.meta.url;
const assert = chai.assert;

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

    async function buildConv(input, name, clip = true, options = undefined) {
      const prefix = './weights/conv_' + name;
      const weightsName = prefix + '_weight.npy';
      const weights =
          await utils.buildConstantFromNpy(builder, new URL(weightsName, url));
      const biasName = prefix + '_bias.npy';
      const bias =
          await utils.buildConstantFromNpy(builder, new URL(biasName, url));
      const conv = builder.add(
          builder.conv2d(input, weights, options),
          builder.reshape(bias, [1, -1, 1, 1]));
      if (clip) {
        return builder.clamp(
          conv,
          {
            minValue: builder.constant(0.),
            maxValue: builder.constant(6.0),
          },
        );
      } else {
        return conv;
      }
    }

    async function buildGemm(input, name) {
      const prefix = './weights/gemm_' + name;
      const weightsName = prefix + '_weight.npy';
      const weights =
          await utils.buildConstantFromNpy(builder, new URL(weightsName, url));
      const biasName = prefix + '_bias.npy';
      const bias =
          await utils.buildConstantFromNpy(builder, new URL(biasName, url));
      const options = {c: bias, bTranspose: true};
      return builder.gemm(input, weights, options);
    }

    async function buildFire(
              input, convNameArray, groups, strides = false, add = true) {
      const conv1x1 = await buildConv(input, convNameArray[0]);
      const options = {
        padding: [1, 1, 1, 1],
        groups: groups,
      };
      if (strides) {
        options.strides = [2, 2];
      }
      const conv3x3 =
          await buildConv(conv1x1, convNameArray[1], true, options);
      const conv1x1NotClip =
          await buildConv(conv3x3, convNameArray[2], false);
      if (add) {
        return builder.add(input, conv1x1NotClip);
      } else {
        return conv1x1NotClip;
      }
    }

    async function buildFireMore(
              input, convNameArray, groupsArrary, strides = true) {
        const out1 = await buildFire(
            input, convNameArray.slice(0, 3), groupsArrary[0], strides, false);
        const out2 =
            await buildFire(out1, convNameArray.slice(3, 6), groupsArrary[1]);
        if (convNameArray.length >= 9) {
          const out3 = await buildFire(
              out2, convNameArray.slice(6, 9), groupsArrary[1]);
          if (convNameArray.length === 12) {
            return await buildFire(
                out3, convNameArray.slice(9, 12), groupsArrary[1]);
          } else {
            return out3;
          }
        } else {
          return out2;
        }
    }

    const data =
        builder.input('input', {type: 'float32', dimensions: [1, 3, 224, 224]});
    const conv0 = await buildConv(
        data, 0, true, {padding: [1, 1, 1, 1], strides: [2, 2]});
    const conv2 = await buildConv(
        conv0, 2, true, {padding: [1, 1, 1, 1], groups: 32});
    const conv4 = await buildConv(conv2, 4, false);
    const add15 =
        await buildFireMore(conv4, [5, 7, 9, 10, 12, 14], [96, 144]);
    const add32 = await buildFireMore(
        add15, [16, 18, 20, 21, 23, 25, 27, 29, 31], [144, 192]);
    const add55 = await buildFireMore(
        add32, [33, 35, 37, 38, 40, 42, 44, 46, 48, 50, 52, 54], [192, 384]);
    const add72 = await buildFireMore(
        add55, [56, 58, 60, 61, 63, 65, 67, 69, 71], [384, 576], false);
    const add89 = await buildFireMore(
        add72, [73, 75, 77, 78, 80, 82, 84, 86, 88], [576, 960]);
    const conv94 =
        await buildFire(add89, [90, 92, 94], 960, false, false);
    const conv95 = await buildConv(conv94, 95, true);
    const pool97 = builder.averagePool2d(conv95);
    const reshape103 = builder.reshape(pool97, [1, -1]);
    const gemm104 = await buildGemm(reshape103, 104);
    graph = await builder.build({gemm104});
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

  async function testMobileNetv2(inputFile, expectedFile) {
    const input = await utils.createTypedArrayFromNpy(new URL(inputFile, url));
    const expected =
        await utils.createTypedArrayFromNpy(new URL(expectedFile, url));
    const outputs = await graph.compute({'input': {data: input}});
    utils.checkShape(outputs.gemm104.dimensions, [1, 1000]);
    utils.checkValue(
        outputs.gemm104.data, expected,
        new utils.AccuracyCriterion(1e-5, 1e-3));
  }

  it('test_data_set_0', async function() {
    await testMobileNetv2(
        './test_data_set_0/input_0.npy', './test_data_set_0/output_0.npy');
  });

  it('test_data_set_1', async function() {
    await testMobileNetv2(
        './test_data_set_1/input_0.npy', './test_data_set_1/output_0.npy');
  });

  it('test_data_set_2', async function() {
    await testMobileNetv2(
        './test_data_set_2/input_0.npy', './test_data_set_2/output_0.npy');
  });
});
