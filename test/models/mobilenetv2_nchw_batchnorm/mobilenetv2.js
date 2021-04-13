'use strict';
import * as utils from '../../utils.js';

const url = import.meta.url;
const assert = chai.assert;

describe('test mobilenetv2 nchw batchnorm', function() {
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

    async function buildConvBatchNorm(
              input, nameIndex, subNameIndex, options = undefined) {
      const subName =
         subNameIndex !== '' ? '_linearbottleneck' + subNameIndex : '';
      let prefix = './weights/mobilenetv20_features' + subName;
      const weightsName = `${prefix}_conv${nameIndex}_weight.npy`;
      const weights =
          await utils.buildConstantFromNpy(builder, new URL(weightsName, url));
      prefix += '_batchnorm' + nameIndex;
      const scaleName = prefix + '_gamma.npy';
      const scale =
          await utils.buildConstantFromNpy(builder, new URL(scaleName, url));
      const biasName = prefix + '_beta.npy';
      const bias =
          await utils.buildConstantFromNpy(builder, new URL(biasName, url));
      const meanName = prefix + '_running_mean.npy';
      const mean = await
          utils.buildConstantFromNpy(builder, new URL(meanName, url));
      const varianceName = prefix + '_running_var.npy';
      const variance =
          await utils.buildConstantFromNpy(builder, new URL(varianceName, url));
      const conv = builder.conv2d(input, weights, options);
      return builder.batchNormalization(
           conv, mean, variance, {scale: scale, bias: bias});
    }

    async function buildFire(input, subNameIndex, options) {
      const batch0 = await buildConvBatchNorm(input, '0', subNameIndex);
      const batch1 = await buildConvBatchNorm(
          builder.relu(batch0), '1', subNameIndex, options);
      return await buildConvBatchNorm(
          builder.relu(batch1), '2', subNameIndex);
    }

    const padding = [1, 1, 1, 1];
    const strides = [2, 2];
    const data = builder.input(
        'input', {type: 'float32', dimensions: [1, 3, 224, 224]});
    const batch0 = await buildConvBatchNorm(
        data, '0', '', {strides, padding});
    const fire0 = await buildFire(
        builder.relu(batch0), '0', {padding, groups: 32});
    const fire1 = await buildFire(
        fire0, '1', {strides, padding, groups: 96});
    const fire2 = await buildFire(fire1, '2', {padding, groups: 144});
    const add0 = builder.add(fire1, fire2);
    const fire3 = await buildFire(
        add0, '3', {strides, padding, groups: 144});
    const fire4 = await buildFire(fire3, '4', {padding, groups: 192});
    const add1 = builder.add(fire3, fire4);
    const fire5 = await buildFire(add1, '5', {padding, groups: 192});
    const add2 = builder.add(add1, fire5);
    const fire6 = await buildFire(add2, '6', {padding, groups: 192});
    const fire7 = await buildFire(fire6, '7', {padding, groups: 384});
    const add3 = builder.add(fire6, fire7);
    const fire8 = await buildFire(add3, '8', {padding, groups: 384});
    const add4 = builder.add(add3, fire8);
    const fire9 = await buildFire(add4, '9', {padding, groups: 384});
    const add5 = builder.add(add4, fire9);
    const fire10 = await buildFire(
        add5, '10', {strides, padding, groups: 384});
    const fire11 = await buildFire(fire10, '11', {padding, groups: 576});
    const add6 = builder.add(fire10, fire11);
    const fire12 = await buildFire(add6, '12', {padding, groups: 576});
    const add7 = builder.add(add6, fire12);
    const fire13 = await buildFire(
        add7, '13', {strides, padding, groups: 576});
    const fire14 = await buildFire(fire13, '14', {padding, groups: 960});
    const add8 = builder.add(fire13, fire14);
    const fire15 = await buildFire(add8, '15', {padding, groups: 960});
    const add9 = builder.add(add8, fire15);
    const fire16 = await buildFire(add9, '16', {padding, groups: 960});
    const batch1 = await buildConvBatchNorm(fire16, '1', '');
    const pool0 = builder.averagePool2d(builder.relu(batch1));
    const weightsConv0 = await utils.buildConstantFromNpy(builder,
        new URL('./weights/mobilenetv20_output_pred_weight.npy', url));
    const conv0 = builder.conv2d(pool0, weightsConv0);
    const reshape0 = builder.reshape(conv0, [1, -1]);
    graph = await builder.build({reshape0});
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
    utils.checkShape(outputs.reshape0.dimensions, [1, 1000]);
    utils.checkValue(
        outputs.reshape0.data, expected,
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
