'use strict';
import * as utils from '../../utils.js';

describe('test squeezenet1.0 nhwc', function() {
  // eslint-disable-next-line no-invalid-this
  this.timeout(0);
  const dirName = typeof __dirname !== 'undefined' ? __dirname :
      './models/squeezenet1.0_nhwc';
  let compiledModel;
  before(async () => {
    const nn = navigator.ml.getNeuralNetworkContext();
    const builder = nn.createModelBuilder();

    async function buildConv(input, name, options = undefined) {
      const prefix = dirName + '/weights/' + name;
      const weights = await utils.buildConstantFromNpy(
          builder, prefix + '_kernel.npy');
      const bias = await utils.buildConstantFromNpy(
          builder, prefix + '_bias.npy');
      if (options !== undefined) {
        options.layout = 'nhwc';
      } else {
        options = {layout: 'nhwc'};
      }
      return builder.relu(builder.add(
          builder.conv2d(input, weights, options),
          builder.reshape(bias, [1, 1, 1, -1])));
    }

    async function buildFire(input, name) {
      const convSqueeze = await buildConv(input, name + '_squeeze');
      const convE1x1 = await buildConv(convSqueeze, name + '_e1x1');
      const convE3x3 = await buildConv(
          convSqueeze, name + '_e3x3', {padding: [1, 1, 1, 1]});
      return builder.concat([convE1x1, convE3x3], 3);
    }

    const placeholder = builder.input('placeholder', {type: 'float32',
        dimensions: [1, 224, 224, 3]});
    const [beginningHeight, endingHeight] =
        utils.computeExplicitPadding(224, 2, 7);
    const [beginningWidth, endingWidth] =
        utils.computeExplicitPadding(224, 2, 7);
    const conv1 = await buildConv(
        placeholder, 'conv1', {
          strides: [2, 2],
          padding: [beginningHeight, endingHeight,
                    beginningWidth, endingWidth]});
    const maxpool1 = builder.maxPool2d(
        conv1, {windowDimensions: [3, 3], strides: [2, 2], layout: 'nhwc'});
    const fire2 = await buildFire(maxpool1, 'fire2');
    const fire3 = await buildFire(fire2, 'fire3');
    const fire4 = await buildFire(fire3, 'fire4');
    const maxpool4 = builder.maxPool2d(
        fire4, {windowDimensions: [3, 3], strides: [2, 2], layout: 'nhwc'});
    const fire5 = await buildFire(maxpool4, 'fire5');
    const fire6 = await buildFire(fire5, 'fire6');
    const fire7 = await buildFire(fire6, 'fire7');
    const fire8 = await buildFire(fire7, 'fire8');
    const maxpool8 = builder.maxPool2d(
        fire8, {windowDimensions: [3, 3], strides: [2, 2], layout: 'nhwc'});
    const fire9 = await buildFire(maxpool8, 'fire9');
    const conv10 = await buildConv(fire9, 'conv10');
    const averagePool2d = builder.averagePool2d(
        conv10, {windowDimensions: [13, 13], layout: 'nhwc'});
    const reshape = builder.reshape(averagePool2d, [1, -1]);
    const softmax = builder.softmax(reshape);
    const model = builder.createModel({softmax});
    compiledModel = await model.compile();
  });

  async function testSqueezeNet(inputFile, expectedFile) {
    const input = await utils.createTypedArrayFromNpy(
      dirName + inputFile);
    const expected = await utils.createTypedArrayFromNpy(
        dirName + expectedFile);
    const outputs = await compiledModel.compute(
        {'placeholder': {buffer: input}});
    utils.checkShape(outputs.softmax.dimensions, [1, 1001]);
    utils.checkValue(outputs.softmax.buffer, expected, 1e-5, 5.0*0.0009765625);
  }

  it('test_data_set_0', async function() {
    await testSqueezeNet('/test_data_set_0/input_0.npy',
                         '/test_data_set_0/output_0.npy');
  });

  it('test_data_set_1', async function() {
    await testSqueezeNet('/test_data_set_1/input_0.npy',
                         '/test_data_set_1/output_0.npy');
  });

  it('test_data_set_2', async function() {
    await testSqueezeNet('/test_data_set_2/input_0.npy',
                         '/test_data_set_2/output_0.npy');
  });
});
