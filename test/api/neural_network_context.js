'use strict';

const expect = chai.expect;

describe('test NeuralNetworkContext', function() {
  it('check navigator.ml', () => {
    expect(navigator.ml).to.be.a('object');
  });

  it('check navigator.ml.getNeuralNetworkContext', () => {
    expect(navigator.ml.getNeuralNetworkContext).to.be.a('function');
  });

  it('getNeuralNetworkContext should return an object', () => {
    expect(navigator.ml.getNeuralNetworkContext()).to.be.a('object');
  });

  it('NeuralNetworkContext should have createModelBuilder method', () => {
    const nn = navigator.ml.getNeuralNetworkContext();
    expect(nn.createModelBuilder).to.be.a('function');
  });

  it('nn.createModelBuilder should return an object', () => {
    const nn = navigator.ml.getNeuralNetworkContext();
    expect(nn.createModelBuilder()).to.be.a('object');
  });
});
