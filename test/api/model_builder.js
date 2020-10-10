'use strict';

const expect = chai.expect;

describe('test ModelBuilder', function() {
  const nn = navigator.ml.getNeuralNetworkContext();
  const builder = nn.createModelBuilder();

  // test input
  it('ModelBuilder should have input method', () => {
    expect(builder.input).to.be.a('function');
  });

  const desc = {type: 'float32', dimensions: [2, 2]};
  it('builder.input should accept a string and an OperandDescriptor', () => {
    expect(builder.input('x', desc)).to.be.a('object');
  });

  it('check operand types for builder.input', () => {
    const dimensions = [2, 2];
    expect(builder.input('1', {type: 'float32', dimensions})).to.be.a('object');
    expect(builder.input('2', {type: 'float16', dimensions})).to.be.a('object');
    expect(builder.input('3', {type: 'int32', dimensions})).to.be.a('object');
    expect(builder.input('4', {type: 'uint32', dimensions})).to.be.a('object');
    expect(builder.input('5', {type: 'int8', dimensions})).to.be.a('object');
    expect(builder.input('6', {type: 'uint8', dimensions})).to.be.a('object');
  });

  it('builder.input should accept scalar operand descriptor', () => {
    expect(builder.input('x', {type: 'float32'})).to.be.a('object');
  });

  it('builder.input should accept scalar operand descriptor', () => {
    expect(builder.input('x', {type: 'float32', dimensions: []}))
        .to.be.a('object');
  });

  it('builder.input should throw for invalid name parameter', () => {
    expect(() => builder.input(1, desc)).to.throw(Error);
    expect(() => builder.input({}, desc)).to.throw(Error);
    expect(() => builder.input(true, desc)).to.throw(Error);
  });

  it('builder.input should throw for invalid desc parameter', () => {
    expect(() => builder.input('x', {})).to.throw(Error);
    expect(() => builder.input('x', 1)).to.throw(Error);
    expect(() => builder.input('x', true)).to.throw(Error);
  });

  it('builder.input should throw for invalid operand type', () => {
    expect(() => builder.input('x', {type: 'float'})).to.throw(Error);
    expect(() => builder.input('x', {type: 'int'})).to.throw(Error);
    expect(() => builder.input('x', {type: {}})).to.throw(Error);
    expect(() => builder.input('x', {type: 1})).to.throw(Error);
    expect(() => builder.input('x', {type: true})).to.throw(Error);
  });

  it('builder.input should throw for invalid dimensions', () => {
    expect(() => builder.input('x', {type: 'float32', dimensions: ['1']}))
        .to.throw(Error);
    expect(() => builder.input('x', {type: 'float32', dimensions: [{}]}))
        .to.throw(Error);
    expect(() => builder.input('x', {type: 'float32', dimensions: [true]}))
        .to.throw(Error);
    expect(() => builder.input('x', {type: 'float32', dimensions: [1.1]}))
        .to.throw(Error);
  });

  // test constant
  it('ModelBuilder should have constant method', () => {
    expect(builder.constant).to.be.a('function');
  });

  it('builder.constant should accept an OperandDescriptor and a value', () => {
    const buffer = new Float32Array(4);
    expect(builder.constant(desc, buffer)).to.be.a('object');
  });

  it('check operand types for builder.constant', () => {
    const dimensions = [2, 2];
    const float32Buffer = new Float32Array(4);
    expect(builder.constant({type: 'float32', dimensions}, float32Buffer))
        .to.be.a('object');
    const uint16Buffer = new Uint16Array(4);
    expect(builder.constant({type: 'float16', dimensions}, uint16Buffer))
        .to.be.a('object');
    const int32Buffer = new Int32Array(4);
    expect(builder.constant({type: 'int32', dimensions}, int32Buffer))
        .to.be.a('object');
    const uint32Buffer = new Uint32Array(4);
    expect(builder.constant({type: 'uint32', dimensions}, uint32Buffer))
        .to.be.a('object');
    const int8Buffer = new Int8Array(4);
    expect(builder.constant({type: 'int8', dimensions}, int8Buffer))
        .to.be.a('object');
    const uint8Buffer = new Uint8Array(4);
    expect(builder.constant({type: 'uint8', dimensions}, uint8Buffer))
        .to.be.a('object');
  });

  it('builder.constant should accept scalar operand descriptor', () => {
    const buffer = new Float32Array(1);
    expect(builder.constant({type: 'float32'}, buffer)).to.be.a('object');
  });

  it('builder.constant should accept scalar operand descriptor', () => {
    const buffer = new Float32Array(1);
    expect(builder.constant({type: 'float32', dimensions: []}, buffer))
        .to.be.a('object');
  });

  it('builder.constant should throw for invalid desc parameter', () => {
    const buffer = new Float32Array(4);
    expect(() => builder.constant({}, buffer)).to.throw(Error);
    expect(() => builder.constant(1, buffer)).to.throw(Error);
    expect(() => builder.constant('', buffer)).to.throw(Error);
    expect(() => builder.constant(true, buffer)).to.throw(Error);
  });

  it('builder.constant should throw for invalid operand type', () => {
    const buffer = new Float32Array(4);
    expect(() => builder.constant({type: 'float'}, buffer)).to.throw(Error);
    expect(() => builder.constant({type: ''}, buffer)).to.throw(Error);
    expect(() => builder.constant({type: 1}, buffer)).to.throw(Error);
    expect(() => builder.constant({type: {}}, buffer)).to.throw(Error);
    expect(() => builder.constant({type: true}, buffer)).to.throw(Error);
  });

  it('builder.constant should throw for invalid dimensions', () => {
    const buffer = new Float32Array(4);
    expect(() => builder.constant({type: 'float32', dimensions: ['']}, buffer))
        .to.throw(Error);
    expect(() => builder.constant({type: 'float32', dimensions: [{}]}, buffer))
        .to.throw(Error);
    expect(
        () => builder.constant({type: 'float32', dimensions: [true]}, buffer))
        .to.throw(Error);
    expect(
        () => builder.constant({type: 'float32', dimensions: [1, 2.2]}, buffer))
        .to.throw(Error);
  });

  it('builder.constant should throw for invalid value type', () => {
    const dimensions = [2, 2];
    expect(() => builder.constant({type: 'float32', dimensions}, [
      1,
      2,
      3,
      4,
    ])).to.throw(Error);
    const float32Buffer = new Float32Array(4);
    expect(() => builder.constant({type: 'float16', dimensions}, float32Buffer))
        .to.throw(Error);
    const uint16Buffer = new Uint16Array(4);
    expect(() => builder.constant({type: 'float32', dimensions}, uint16Buffer))
        .to.throw(Error);
    const int32Buffer = new Int32Array(4);
    expect(() => builder.constant({type: 'uint32', dimensions}, int32Buffer))
        .to.throw(Error);
    const uint32Buffer = new Uint32Array(4);
    expect(() => builder.constant({type: 'int32', dimensions}, uint32Buffer))
        .to.throw(Error);
    const int8Buffer = new Int8Array(4);
    expect(() => builder.constant({type: 'uint8', dimensions}, int8Buffer))
        .to.throw(Error);
    const uint8Buffer = new Uint8Array(4);
    expect(() => builder.constant({type: 'int8', dimensions}, uint8Buffer))
        .to.throw(Error);
  });

  it('builder.constant should throw for invalid value length', () => {
    const buffer = new Float32Array(4);
    expect(() => builder.constant({type: 'float32'}, buffer)).to.throw(Error);
    expect(() => builder.constant({type: 'float32', dimensions: []}, buffer))
        .to.throw(Error);
    expect(() => builder.constant({type: 'float32', dimensions: [2]}, buffer))
        .to.throw(Error);
    expect(
        () => builder.constant({type: 'float32', dimensions: [2, 3]}, buffer))
        .to.throw(Error);
    expect(
        () =>
            builder.constant({type: 'float32', dimensions: [2, 2, 2]}, buffer))
        .to.throw(Error);
  });

  it('builder.constant should accept a single-value', () => {
    expect(builder.constant(1)).to.be.a('object');
    expect(builder.constant(1.0)).to.be.a('object');
    expect(builder.constant(0.1)).to.be.a('object');
  });

  it('builder.constant should accept a single-value and an operand type',
     () => {
       expect(builder.constant(1.0, 'float32')).to.be.a('object');
       expect(builder.constant(1.0, 'float16')).to.be.a('object');
       expect(builder.constant(1, 'int32')).to.be.a('object');
       expect(builder.constant(1, 'uint32')).to.be.a('object');
       expect(builder.constant(1, 'int8')).to.be.a('object');
       expect(builder.constant(1, 'uint8')).to.be.a('object');
     });

  it('builder.constant should throw for invalid value', () => {
    expect(() => builder.constant('1')).to.throw(Error);
    expect(() => builder.constant({})).to.throw(Error);
    expect(() => builder.constant(true)).to.throw(Error);
  });

  it('builder.constant should throw for invalid type', () => {
    expect(() => builder.constant(1.0, 'float')).to.throw(Error);
    expect(() => builder.constant(1.0, '')).to.throw(Error);
    expect(() => builder.constant(1.0, {})).to.throw(Error);
    expect(() => builder.constant(1.0, true)).to.throw(Error);
  });

  it('builder.constant should throw for invalid value according to type',
     () => {
       expect(() => builder.constant(1.1, 'int32')).to.throw(Error);
       expect(() => builder.constant(-1, 'uint32')).to.throw(Error);
       expect(() => builder.constant(1000, 'int8')).to.throw(Error);
       expect(() => builder.constant(-1, 'uint8')).to.throw(Error);
     });

  // test createModel
  it('builder.createModel should be a function', () => {
    expect(builder.createModel).to.be.a('function');
  });

  it('builder.createModel should return an object', () => {
    const a = builder.input('a', desc);
    const b = builder.input('b', desc);
    const c = builder.matmul(a, b);
    expect(builder.createModel({c})).to.be.a('object');
  });

  it('builder.createModel should throw for invalid outputs', () => {
    expect(() => builder.createModel()).to.throw(Error);
    expect(() => builder.createModel({})).to.throw(Error);
    expect(() => builder.createModel({'a': 1})).to.throw(Error);
    expect(() => builder.createModel({'a': {}})).to.throw(Error);
  });
});
