'use strict';
import * as utils from '../../utils.js';

describe('test squeezenet1.1', function() {
  const dirName = typeof __dirname !== 'undefined' ? __dirname :
      './models/squeezenet1.1';
  let compiledModel;
  before(async () => {
    const nn = navigator.ml.getNeuralNetworkContext();
    const builder = nn.createModelBuilder();
    const data = builder.input('data', {type: 'float32', dimensions: [1, 3, 224, 224]});
    const squeezenet0_conv0_weight = await utils.buildConstantFromNpy(
        builder, dirName + '/weights/squeezenet0_conv0_weight.npy');
    const squeezenet0_conv0_bias = await utils.buildConstantFromNpy(
        builder, dirName + '/weights/squeezenet0_conv0_bias.npy');
    const squeezenet0_conv0_fwd = builder.add(
        builder.conv2d(data, squeezenet0_conv0_weight, {strides: [2, 2]}),
        builder.reshape(squeezenet0_conv0_bias, [1, 64, 1, 1]));
    const squeezenet0_relu0_fwd = builder.relu(squeezenet0_conv0_fwd);
    const squeezenet0_pool0_fwd = builder.maxPool2d(
        squeezenet0_relu0_fwd,
        {windowDimensions: [3, 3], strides: [2, 2]});
    const squeezenet0_conv1_weight = await utils.buildConstantFromNpy(
        builder, dirName + '/weights/squeezenet0_conv1_weight.npy');
    const squeezenet0_conv1_bias = await utils.buildConstantFromNpy(
        builder, dirName + '/weights/squeezenet0_conv1_bias.npy');
    const squeezenet0_conv1_fwd = builder.add(
        builder.conv2d(squeezenet0_pool0_fwd, squeezenet0_conv1_weight),
        builder.reshape(squeezenet0_conv1_bias, [1, 16, 1, 1]));
    const squeezenet0_relu1_fwd = builder.relu(squeezenet0_conv1_fwd);
    const squeezenet0_conv2_weight = await utils.buildConstantFromNpy(
        builder, dirName + '/weights/squeezenet0_conv2_weight.npy');
    const squeezenet0_conv2_bias = await utils.buildConstantFromNpy(
        builder, dirName + '/weights/squeezenet0_conv2_bias.npy');
    const squeezenet0_conv2_fwd = builder.add(
        builder.conv2d(squeezenet0_relu1_fwd, squeezenet0_conv2_weight),
        builder.reshape(squeezenet0_conv2_bias, [1, 64, 1, 1]));
    const squeezenet0_relu2_fwd = builder.relu(squeezenet0_conv2_fwd);
    const squeezenet0_conv3_weight = await utils.buildConstantFromNpy(
        builder, dirName + '/weights/squeezenet0_conv3_weight.npy');
    const squeezenet0_conv3_bias = await utils.buildConstantFromNpy(
        builder, dirName + '/weights/squeezenet0_conv3_bias.npy');
    const squeezenet0_conv3_fwd = builder.add(
        builder.conv2d(squeezenet0_relu1_fwd, squeezenet0_conv3_weight,
                       {padding: [1, 1, 1, 1]}),
        builder.reshape(squeezenet0_conv3_bias, [1, 64, 1, 1]));
    const squeezenet0_relu3_fwd = builder.relu(squeezenet0_conv3_fwd);
    const squeezenet0_concat0 = builder.concat(
        [squeezenet0_relu2_fwd, squeezenet0_relu3_fwd], 1);
    const model = builder.createModel(
        {squeezenet0_flatten0_reshape0: squeezenet0_concat0});
    compiledModel = await model.compile();
  });

  it('test_data_set_0', async function() {
    const input = await utils.createTypedArrayFromNpy(
        dirName + '/test_data_set_0/input_0.npy');
    const expected = await utils.createTypedArrayFromNpy(
        dirName + '/test_data_set_0/output_0.npy');
    const outputs = await compiledModel.compute({'data': {buffer: input}});
    console.log(outputs.squeezenet0_flatten0_reshape0.dimensions);
    console.log(outputs.squeezenet0_flatten0_reshape0.buffer);
    // utils.checkShape(outputs.y.dimensions, [3, 4]);
    // utils.checkValue(outputs.y.buffer, expected);
  });
});
