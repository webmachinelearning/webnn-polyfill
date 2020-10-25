'use strict';

const expect = chai.expect;

describe('test NeuralNetworkContext', function() {
  it('navigator.ml should be a ML', () => {
    expect(navigator.ml).to.be.an.instanceof(ML);
  });

  it('ml.getNeuralNetworkContext should be a function', () => {
    expect(navigator.ml.getNeuralNetworkContext).to.be.a('function');
  });

  it('ml.getNeuralNetworkContext should return a NeuralNetworkContext', () => {
    expect(navigator.ml.getNeuralNetworkContext())
            .to.be.an.instanceof(NeuralNetworkContext);
  });

  it('NeuralNetworkContext should have createModelBuilder method', () => {
    const nn = navigator.ml.getNeuralNetworkContext();
    expect(nn.createModelBuilder).to.be.a('function');
  });

  it('nn.createModelBuilder should return a ModelBuilder', () => {
    const nn = navigator.ml.getNeuralNetworkContext();
    expect(nn.createModelBuilder()).to.be.an.instanceof(ModelBuilder);
  });
});
