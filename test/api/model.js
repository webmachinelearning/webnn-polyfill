'use strict';

const expect = chai.expect;
const assert = chai.assert;

describe('test Model', function() {
  const nn = navigator.ml.getNeuralNetworkContext();
  const builder = nn.createModelBuilder();
  const desc = {type: 'float32', dimensions: [2, 2]};
  const a = builder.input('a', desc);
  const b = builder.input('b', desc);
  const c = builder.matmul(a, b);
  const model = builder.createModel({c});

  it('Model should have compile method', () => {
    expect(model.compile).to.be.a('function');
  });

  it('Model.compile should return a promise', async () => {
    expect(await model.compile()).to.be.an.instanceof(Compilation);
  });

  it('Model.compile should support CompilationOptions', async () => {
    expect(await model.compile({})).to.be.an.instanceof(Compilation);
    expect(await model.compile({
      powerPreference: 'default',
    })).to.be.an.instanceof(Compilation);
    expect(await model.compile({
      powerPreference: 'low-power',
    })).to.be.an.instanceof(Compilation);
    expect(await model.compile({
      powerPreference: 'high-performance',
    })).to.be.an.instanceof(Compilation);
  });

  it('Model.compile should throw for invalid options', async () => {
    try {
      await model.compile('invalid');
      assert.fail();
    } catch (err) {
      expect(err).not.to.be.an.instanceof(chai.AssertionError);
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('Model.compile should throw for invalid power preference', async () => {
    try {
      await model.compile({powerPreference: 'invalid'});
      assert.fail();
    } catch (err) {
      expect(err).not.to.be.an.instanceof(chai.AssertionError);
      expect(err).to.be.an.instanceof(Error);
    }
  });
});
