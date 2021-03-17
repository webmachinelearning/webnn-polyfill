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
    expect(builder.input('x', desc)).to.be.an.instanceof(Operand);
  });

  it('check operand types for builder.input', () => {
    const dimensions = [2, 2];
    expect(builder.input('1', {type: 'float32', dimensions}))
            .to.be.an.instanceof(Operand);
    expect(builder.input('2', {type: 'float16', dimensions}))
            .to.be.an.instanceof(Operand);
    expect(builder.input('3', {type: 'int32', dimensions}))
            .to.be.an.instanceof(Operand);
    expect(builder.input('4', {type: 'uint32', dimensions}))
            .to.be.an.instanceof(Operand);
    expect(builder.input('5', {type: 'int8', dimensions}))
            .to.be.an.instanceof(Operand);
    expect(builder.input('6', {type: 'uint8', dimensions}))
            .to.be.an.instanceof(Operand);
  });

  it('builder.input should accept scalar operand descriptor', () => {
    expect(builder.input('x', {type: 'float32'})).to.be.an.instanceof(Operand);
  });

  it('builder.input should accept scalar operand descriptor', () => {
    expect(builder.input('x', {type: 'float32', dimensions: []}))
            .to.be.an.instanceof(Operand);
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
    expect(builder.constant(desc, buffer)).to.be.an.instanceof(Operand);
  });

  it('check operand types for builder.constant', () => {
    const dimensions = [2, 2];
    const float32Buffer = new Float32Array(4);
    expect(builder.constant({type: 'float32', dimensions}, float32Buffer))
            .to.be.an.instanceof(Operand);
    const uint16Buffer = new Uint16Array(4);
    expect(builder.constant({type: 'float16', dimensions}, uint16Buffer))
            .to.be.an.instanceof(Operand);
    const int32Buffer = new Int32Array(4);
    expect(builder.constant({type: 'int32', dimensions}, int32Buffer))
            .to.be.an.instanceof(Operand);
    const uint32Buffer = new Uint32Array(4);
    expect(builder.constant({type: 'uint32', dimensions}, uint32Buffer))
            .to.be.an.instanceof(Operand);
    const int8Buffer = new Int8Array(4);
    expect(builder.constant({type: 'int8', dimensions}, int8Buffer))
            .to.be.an.instanceof(Operand);
    const uint8Buffer = new Uint8Array(4);
    expect(builder.constant({type: 'uint8', dimensions}, uint8Buffer))
            .to.be.an.instanceof(Operand);
  });

  it('builder.constant should accept scalar operand descriptor', () => {
    const buffer = new Float32Array(1);
    expect(builder.constant({type: 'float32'}, buffer))
            .to.be.an.instanceof(Operand);
  });

  it('builder.constant should accept scalar operand descriptor', () => {
    const buffer = new Float32Array(1);
    expect(builder.constant({type: 'float32', dimensions: []}, buffer))
            .to.be.an.instanceof(Operand);
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
    expect(builder.constant(1)).to.be.an.instanceof(Operand);
    expect(builder.constant(1.0)).to.be.an.instanceof(Operand);
    expect(builder.constant(0.1)).to.be.an.instanceof(Operand);
  });

  it('builder.constant should accept a single-value and an operand type',
     () => {
       expect(builder.constant(1.0, 'float32')).to.be.an.instanceof(Operand);
       expect(builder.constant(1.0, 'float16')).to.be.an.instanceof(Operand);
       expect(builder.constant(1, 'int32')).to.be.an.instanceof(Operand);
       expect(builder.constant(1, 'uint32')).to.be.an.instanceof(Operand);
       expect(builder.constant(1, 'int8')).to.be.an.instanceof(Operand);
       expect(builder.constant(1, 'uint8')).to.be.an.instanceof(Operand);
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

  // test add
  it('builder should have add method', () => {
    expect(builder.add).to.be.a('function');
  });

  it('builder.add should return an operand', () => {
    const a = builder.input('a', desc);
    const b = builder.input('b', desc);
    expect(builder.add(a, b)).to.be.an.instanceof(Operand);
  });

  it('builder.add should throw for invalid parameters', () => {
    expect(() => builder.add(1, 2)).to.throw(Error);
    expect(() => builder.add('a', 'b')).to.throw(Error);
    expect(() => builder.add({}, {})).to.throw(Error);
    const a = builder.input('a', desc);
    expect(() => builder.add(a)).to.throw(Error);
  });

  // test mul
  it('builder should have mul method', () => {
    expect(builder.mul).to.be.a('function');
  });

  it('builder.mul should return an operand', () => {
    const a = builder.input('a', desc);
    const b = builder.input('b', desc);
    expect(builder.mul(a, b)).to.be.an.instanceof(Operand);
  });

  it('builder.mul should throw for invalid parameters', () => {
    expect(() => builder.mul(1, 2)).to.throw(Error);
    expect(() => builder.mul('a', 'b')).to.throw(Error);
    expect(() => builder.mul({}, {})).to.throw(Error);
    const a = builder.input('a', desc);
    expect(() => builder.mul(a)).to.throw(Error);
  });

  // test conv2d
  it('builder should have conv2d method', () => {
    expect(builder.conv2d).to.be.a('function');
  });

  it('builder.conv2d should return an operand', () => {
    const input =
        builder.input('input', {type: 'float32', dimensions: [1, 1, 5, 5]});
    const filter = builder.constant(
        {type: 'float32', dimensions: [1, 1, 3, 3]},
        new Float32Array(9).fill(1));
    expect(builder.conv2d(input, filter)).to.be.an.instanceof(Operand);
  });

  it('builder.conv2d should throw for invalid parameters', () => {
    expect(() => builder.conv2d(1, 2)).to.throw(Error);
    expect(() => builder.conv2d('a', 'b')).to.throw(Error);
    expect(() => builder.conv2d({}, {})).to.throw(Error);
    const input =
        builder.input('input', {type: 'float32', dimensions: [1, 1, 5, 5]});
    const filter = builder.constant(
        {type: 'float32', dimensions: [1, 1, 3, 3]},
        new Float32Array(9).fill(1));
    expect(() => builder.conv2d(input)).to.throw(Error);
    expect(() => builder.conv2d(input, filter, {padding: 0})).to.throw(Error);
    expect(() => builder.conv2d(input, filter, {padding: []})).to.throw(Error);
    expect(() => builder.conv2d(input, filter, {
      padding: [0, 0],
    })).to.throw(Error);
    expect(() => builder.conv2d(input, filter, {
      padding: [0, 0, 0, 0],
      strides: 1,
    })).to.throw(Error);
    expect(() => builder.conv2d(input, filter, {
      padding: [0, 0, 0, 0],
      strides: [],
    })).to.throw(Error);
    expect(() => builder.conv2d(input, filter, {
      padding: [0, 0, 0, 0],
      strides: [1],
    })).to.throw(Error);
    expect(() => builder.conv2d(input, filter, {
      padding: [0, 0, 0, 0],
      strides: [1, 1],
      dilations: 1,
    })).to.throw(Error);
    expect(() => builder.conv2d(input, filter, {
      padding: [0, 0, 0, 0],
      strides: [1, 1],
      dilations: [],
    })).to.throw(Error);
    expect(() => builder.conv2d(input, filter, {
      padding: [0, 0, 0, 0],
      strides: [1, 1],
      dilations: [1],
    })).to.throw(Error);
    expect(() => builder.conv2d(input, filter, {
      padding: [0, 0, 0, 0],
      strides: [1, 1],
      dilations: [1, 1],
      groups: [1],
    })).to.throw(Error);
    expect(() => builder.conv2d(input, filter, {
      padding: [0, 0, 0, 0],
      strides: [1, 1],
      dilations: [1, 1],
      groups: 1,
      inputLayout: 'abcd',
    })).to.throw(Error);
  });

  // test matmul
  it('builder should have matmul method', () => {
    expect(builder.matmul).to.be.a('function');
  });

  it('builder.matmul should return an operand', () => {
    const a = builder.input('a', {type: 'float32', dimensions: [3, 4]});
    const b = builder.input('b', {type: 'float32', dimensions: [4, 3]});
    expect(builder.matmul(a, b)).to.be.an.instanceof(Operand);
  });

  it('builder.matmul should throw for invalid parameters', () => {
    expect(() => builder.matmul(1, 2)).to.throw(Error);
    expect(() => builder.matmul('a', 'b')).to.throw(Error);
    expect(() => builder.matmul({}, {})).to.throw(Error);
    const a = builder.input('a', {type: 'float32', dimensions: [3, 4]});
    expect(() => builder.matmul(a)).to.throw(Error);
  });

  // test averagePool2d
  it('builder should have averagePool2d method', () => {
    expect(builder.averagePool2d).to.be.a('function');
  });

  it('builder.averagePool2d should return an operand', () => {
    const input =
        builder.input('input', {type: 'float32', dimensions: [1, 1, 5, 5]});
    expect(builder.averagePool2d(input, {
      windowDimensions: [2, 2],
    })).to.be.an.instanceof(Operand);
  });

  it('builder.averagePool2d should throw for invalid parameters', () => {
    expect(() => builder.averagePool2d(1, 2)).to.throw(Error);
    expect(() => builder.averagePool2d('a', 'b')).to.throw(Error);
    expect(() => builder.averagePool2d({}, {})).to.throw(Error);
    const input =
        builder.input('input', {type: 'float32', dimensions: [1, 1, 5, 5]});
    expect(() => builder.averagePool2d(input, {
      windowDimensions: [],
    })).to.throw(Error);
    expect(() => builder.averagePool2d(input, {
      windowDimensions: [2, 2],
      padding: 0,
    })).to.throw(Error);
    expect(() => builder.averagePool2d(input, {
      windowDimensions: [2, 2],
      padding: [],
    })).to.throw(Error);
    expect(() => builder.averagePool2d(input, {
      windowDimensions: [2, 2],
      padding: [0, 0],
    })).to.throw(Error);
    expect(() => builder.averagePool2d(input, {
      windowDimensions: [2, 2],
      padding: [0, 0, 0, 0],
      strides: 1,
    })).to.throw(Error);
    expect(() => builder.averagePool2d(input, {
      windowDimensions: [2, 2],
      padding: [0, 0, 0, 0],
      strides: [],
    })).to.throw(Error);
    expect(() => builder.averagePool2d(input, {
      windowDimensions: [2, 2],
      padding: [0, 0, 0, 0],
      strides: [1],
    })).to.throw(Error);
    expect(() => builder.averagePool2d(input, {
      windowDimensions: [2, 2],
      padding: [0, 0, 0, 0],
      strides: [1, 1],
      dilations: 1,
    })).to.throw(Error);
    expect(() => builder.averagePool2d(input, {
      windowDimensions: [2, 2],
      padding: [0, 0, 0, 0],
      strides: [1, 1],
      dilations: [],
    })).to.throw(Error);
    expect(() => builder.averagePool2d(input, {
      windowDimensions: [2, 2],
      padding: [0, 0, 0, 0],
      strides: [1, 1],
      dilations: [1],
    })).to.throw(Error);
    expect(() => builder.averagePool2d(input, {
      windowDimensions: [2, 2],
      padding: [0, 0, 0, 0],
      strides: [1, 1],
      dilations: [1, 1],
      layout: 'abcd',
    })).to.throw(Error);
  });

  // test maxPool2d
  it('builder should have maxPool2d method', () => {
    expect(builder.maxPool2d).to.be.a('function');
  });

  it('builder.maxPool2d should return an operand', () => {
    const input =
        builder.input('input', {type: 'float32', dimensions: [1, 1, 5, 5]});
    expect(builder.maxPool2d(input, {
      windowDimensions: [2, 2],
    })).to.be.an.instanceof(Operand);
  });

  it('builder.maxPool2d should throw for invalid parameters', () => {
    expect(() => builder.maxPool2d(1, 2)).to.throw(Error);
    expect(() => builder.maxPool2d('a', 'b')).to.throw(Error);
    expect(() => builder.maxPool2d({}, {})).to.throw(Error);
    const input =
        builder.input('input', {type: 'float32', dimensions: [1, 1, 5, 5]});
    expect(() => builder.maxPool2d(input, {
      windowDimensions: [],
    })).to.throw(Error);
    expect(
        () => builder.maxPool2d(input, {windowDimensions: [2, 2], padding: 0}))
        .to.throw(Error);
    expect(
        () => builder.maxPool2d(input, {windowDimensions: [2, 2], padding: []}))
        .to.throw(Error);
    expect(() => builder.maxPool2d(input, {
      windowDimensions: [2, 2],
      padding: [0, 0],
    })).to.throw(Error);
    expect(() => builder.maxPool2d(input, {
      windowDimensions: [2, 2],
      padding: [0, 0, 0, 0],
      strides: 1,
    })).to.throw(Error);
    expect(() => builder.maxPool2d(input, {
      windowDimensions: [2, 2],
      padding: [0, 0, 0, 0],
      strides: [],
    })).to.throw(Error);
    expect(() => builder.maxPool2d(input, {
      windowDimensions: [2, 2],
      padding: [0, 0, 0, 0],
      strides: [1],
    })).to.throw(Error);
    expect(() => builder.maxPool2d(input, {
      windowDimensions: [2, 2],
      padding: [0, 0, 0, 0],
      strides: [1, 1],
      dilations: 1,
    })).to.throw(Error);
    expect(() => builder.maxPool2d(input, {
      windowDimensions: [2, 2],
      padding: [0, 0, 0, 0],
      strides: [1, 1],
      dilations: [],
    })).to.throw(Error);
    expect(() => builder.maxPool2d(input, {
      windowDimensions: [2, 2],
      padding: [0, 0, 0, 0],
      strides: [1, 1],
      dilations: [1],
    })).to.throw(Error);
    expect(() => builder.maxPool2d(input, {
      windowDimensions: [2, 2],
      padding: [0, 0, 0, 0],
      strides: [1, 1],
      dilations: [1, 1],
      layout: 'abcd',
    })).to.throw(Error);
  });

  // test relu
  it('builder should have relu method', () => {
    expect(builder.relu).to.be.a('function');
  });

  it('builder.relu should return an operand', () => {
    const x = builder.input('x', {type: 'float32', dimensions: [4, 4]});
    expect(builder.relu(x)).to.be.an.instanceof(Operand);
  });

  it('builder.relu should throw for invalid parameters', () => {
    expect(() => builder.relu(1)).to.throw(Error);
    expect(() => builder.relu('x')).to.throw(Error);
    expect(() => builder.relu({})).to.throw(Error);
  });

  // test reshape
  it('builder should have reshape method', () => {
    expect(builder.reshape).to.be.a('function');
  });

  it('builder.reshape should return an operand', () => {
    const x = builder.input('x', {type: 'float32', dimensions: [4, 4]});
    expect(builder.reshape(x, [1, -1])).to.be.an.instanceof(Operand);
  });

  it('builder.reshape should throw for invalid parameters', () => {
    expect(() => builder.reshape(1, [1, -1])).to.throw(Error);
    expect(() => builder.reshape('x', [1, -1])).to.throw(Error);
    expect(() => builder.reshape({}, [1, -1])).to.throw(Error);
    const x = builder.input('x', {type: 'float32', dimensions: [4, 4]});
    expect(() => builder.reshape(x)).to.throw(Error);
    expect(() => builder.reshape(x, [])).to.throw(Error);
    expect(() => builder.reshape(x, ['1', '-1'])).to.throw(Error);
  });

  // test transpose
  it('builder should have transpose method', () => {
    expect(builder.transpose).to.be.a('function');
  });

  it('builder.transpose should return an operand', () => {
    const x = builder.input('x', {type: 'float32', dimensions: [3, 4]});
    expect(builder.transpose(x)).to.be.an.instanceof(Operand);
  });

  it('builder.transpose should throw for invalid parameters', () => {
    expect(() => builder.transpose(1)).to.throw(Error);
    expect(() => builder.transpose('x')).to.throw(Error);
    expect(() => builder.transpose({})).to.throw(Error);
    const x = builder.input('x', {type: 'float32', dimensions: [3, 4]});
    expect(() => builder.transpose(x, {permutation: []})).to.throw(Error);
    expect(() => builder.transpose(x, {permutation: [{}]})).to.throw(Error);
    expect(() => builder.transpose(x, {
      permutation: ['1', '-1'],
    })).to.throw(Error);
  });

  // test softmax
  it('builder should have softmax method', () => {
    expect(builder.softmax).to.be.a('function');
  });

  it('builder.softmax should return an operand', () => {
    const x = builder.input('x', {type: 'float32', dimensions: [4, 4]});
    expect(builder.softmax(x)).to.be.an.instanceof(Operand);
  });

  it('builder.softmax should throw for invalid parameters', () => {
    expect(() => builder.softmax(1)).to.throw(Error);
    expect(() => builder.softmax('x')).to.throw(Error);
    expect(() => builder.softmax({})).to.throw(Error);
  });

  // test createModel
  it('builder.createModel should be a function', () => {
    expect(builder.createModel).to.be.a('function');
  });

  it('builder.createModel should return a Model', () => {
    const a = builder.input('a', desc);
    const b = builder.input('b', desc);
    const c = builder.matmul(a, b);
    expect(builder.createModel({c})).to.be.an.instanceof(Model);
  });

  it('builder.createModel should throw for invalid parameters', () => {
    expect(() => builder.createModel()).to.throw(Error);
    expect(() => builder.createModel({})).to.throw(Error);
    expect(() => builder.createModel({'a': 1})).to.throw(Error);
    expect(() => builder.createModel({'a': {}})).to.throw(Error);
  });

  it('builder should throw for cross builders inputs', () => {
    const builder2 = nn.createModelBuilder();
    const a = builder.input('a', desc);
    const b = builder2.input('b', desc);
    expect(() => builder.matmul(a, b)).to.throw(Error);
    expect(() => builder2.matmul(a, b)).to.throw(Error);
  });

  it('builder.createModel should throw for duplicated inputs', () => {
    const a = builder.input('a', desc);
    const b = builder.input('a', desc);
    const c = builder.matmul(a, b);
    expect(() => builder.createModel({c})).to.throw(Error);
  });
});
