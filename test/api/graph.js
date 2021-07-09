'use strict';

const expect = chai.expect;
const assert = chai.assert;

import * as utils from '../utils.js';

describe('test MLGraph', function() {
  const context = navigator.ml.createContext();
  const builder = new MLGraphBuilder(context);
  const desc = {type: 'float32', dimensions: [2, 2]};
  const a = builder.input('a', desc);
  const b = builder.input('b', desc);
  const c = builder.matmul(a, b);
  const d = builder.constant(
      {type: 'float32', dimensions: [2, 2]}, new Float32Array(4).fill(1));
  const e = builder.add(c, d);
  const bufferA = new Float32Array(4).fill(1);
  const bufferB = new Float32Array(4).fill(1);
  const bufferC = new Float32Array(4);
  const bufferE = new Float32Array(4);
  const expectedC = [2, 2, 2, 2];
  const expectedE = [3, 3, 3, 3];

  it('MLGraph should have compute method', () => {
    const graph = builder.build({c});
    expect(graph.compute).to.be.a('function');
  });

  it('MLGraph.compute should accept inputs and outputs', () => {
    const graph = builder.build({c});
    const inputs = {a: bufferA, b: bufferB};
    const outputs = {c: bufferC};
    graph.compute(inputs, outputs);
    utils.checkValue(outputs.c, expectedC);
  });

  it('MLGraph.compute should support multiple outputs', () => {
    const graph = builder.build({c, e});
    const inputs = {a: bufferA, b: bufferB};
    const outputs = {c: bufferC, e: bufferE};
    graph.compute(inputs, outputs);
    utils.checkValue(outputs.c, expectedC);
    utils.checkValue(outputs.e, expectedE);
  });

  it('MLGraph.compute should support specified outputs', () => {
    const graph = builder.build({c, e});
    const inputs = {a: bufferA, b: bufferB};
    let outputs = {c: bufferC};
    graph.compute(inputs, outputs);
    utils.checkValue(outputs.c, expectedC);
    expect(outputs).not.to.have.property('e');
    outputs = {e: bufferE};
    graph.compute(inputs, outputs);
    utils.checkValue(outputs.e, expectedE);
    expect(outputs).not.to.have.property('c');
  });

  const descX = {type: 'float32', dimensions: [-1, 2]};
  const descY = {type: 'float32', dimensions: [2, -1]};
  const x = builder.input('x', descX);
  const y = builder.input('y', descY);
  const z = builder.matmul(x, y);
  it('MLGraph.compute should support inputs with specified shape', () => {
    const graph = builder.build({z});
    const shapeX = [3, 2];
    const bufferX = new Float32Array(utils.sizeOfShape(shapeX)).fill(1);
    const shapeY = [2, 4];
    const bufferY = new Float32Array(utils.sizeOfShape(shapeY)).fill(1);
    const inputs = {
      x: {resource: bufferX, dimensions: shapeX},
      y: {resource: bufferY, dimensions: shapeY},
    };
    const shapeZ = [shapeX[0], shapeY[1]];
    const outputs = {z: new Float32Array(utils.sizeOfShape(shapeZ))};
    graph.compute(inputs, outputs);
    const expectedZ = new Array(utils.sizeOfShape(shapeZ)).fill(2);
    utils.checkValue(outputs.z, expectedZ);
  });

  it('MLGraph.compute should throw for non inputs', () => {
    const graph = builder.build({c});
    try {
      graph.compute();
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLGraph.compute should throw for empty inputs', () => {
    const graph = builder.build({c});
    try {
      graph.compute({}, {c: bufferC});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLGraph.compute should throw for non outputs', () => {
    const graph = builder.build({c});
    try {
      graph.compute({a: bufferA, b: bufferB});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLGraph.compute should throw for invalid input name', () => {
    const graph = builder.build({c});
    try {
      graph.compute({x: bufferA}, {c: bufferC});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLGraph.compute should throw for missing input', () => {
    const graph = builder.build({c});
    try {
      graph.compute({a: bufferA}, {c: bufferC});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLGraph.compute should throw for no input resource', () => {
    const graph = builder.build({c});
    try {
      graph.compute({a: {}, b: {}}, {c: bufferC});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLGraph.compute should throw for invalid input data', () => {
    const graph = builder.build({c});
    try {
      graph.compute({a: 1, b: 2});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLGraph.compute should throw for invalid input dimensions', () => {
    const graph = builder.build({c});
    try {
      graph.compute(
          {
            a: {resource: bufferA, dimensions: [2]},
            b: {resource: bufferB, dimensions: [2]},
          },
          {c: bufferC});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLGraph.compute should throw for no dimensions for dynamic shape', () => {
    const graph = builder.build({z});
    const shapeX = [3, 2];
    const bufferX = new Float32Array(utils.sizeOfShape(shapeX)).fill(1);
    const shapeY = [2, 4];
    const bufferY = new Float32Array(utils.sizeOfShape(shapeY)).fill(1);
    const inputs = {x: {resource: bufferX}, y: {resource: bufferY}};
    const shapeZ = [shapeX[0], shapeY[1]];
    const outputs = {z: new Float32Array(utils.sizeOfShape(shapeZ))};
    try {
      graph.compute(inputs, outputs);
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLGraph.compute should throw for invalid output name', () => {
    const graph = builder.build({c});
    try {
      const inputs = {a: bufferA, b: bufferB};
      const bufferC = new Float32Array(4);
      const outputs = {z: bufferC};
      graph.compute(inputs, outputs);
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLGraph.compute should throw for invalid output data', () => {
    const graph = builder.build({c});
    try {
      const inputs = {a: bufferA, b: bufferB};
      const outputs = {c: []};
      graph.compute(inputs, outputs);
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLGraph.compute should throw for invalid output data length', () => {
    const graph = builder.build({c});
    try {
      const inputs = {a: bufferA, b: bufferB};
      const bufferC = new Float32Array(1);
      const outputs = {c: bufferC};
      graph.compute(inputs, outputs);
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLGraph should be immutable after creation', () => {
    const builder = new MLGraphBuilder(context);
    const desc = {type: 'float32', dimensions: [2, 2]};
    const a = builder.input('a', desc);
    const bufferB = new Float32Array(4).fill(1);
    let b = builder.constant({type: 'float32', dimensions: [2, 2]}, bufferB);
    const c = builder.matmul(a, b);
    const bufferA = new Float32Array(4).fill(1);
    const expectedC = [2, 2, 2, 2];
    const graph = builder.build({c});
    let inputs = {a: bufferA};
    const outputs = {c: bufferC};
    graph.compute(inputs, outputs);
    utils.checkValue(outputs.c, expectedC);

    // Change data of constant b should not impact graph compute.
    bufferB.set(new Array(4).fill(2));
    graph.compute(inputs, outputs);
    utils.checkValue(outputs.c, expectedC);

    // Replace b with a new constant should not impact graph compute.
    b = builder.constant({type: 'float32', dimensions: [2, 2]}, bufferB);
    graph.compute(inputs, outputs);
    utils.checkValue(outputs.c, expectedC);

    // Change opearnd type of b should not impact graph compute.
    b = builder.input('b', desc);
    graph.compute(inputs, outputs);
    utils.checkValue(outputs.c, expectedC);

    // Create new model with new b.
    const graph2 = builder.build({'c': builder.matmul(a, b)});
    inputs = {'a': bufferA, 'b': bufferB};
    graph2.compute(inputs, outputs);
    utils.checkValue(outputs.c, [4, 4, 4, 4]);
  });

  it('MLGraph should not leak memory', async () => {
    // Only run this test for polyfill.
    if (typeof _tfengine !== 'undefined') {
      const beforeNumBytes = _tfengine.memory().numBytes;
      const beforeNumTensors = _tfengine.memory().numTensors;

      // Run gru modele which is a complex graph
      const builder = new MLGraphBuilder(context);
      const steps = 2;
      const numDirections = 1;
      const batchSize = 3;
      const inputSize = 3;
      const hiddenSize = 5;
      const input = builder.input(
          'input',
          {type: 'float32', dimensions: [steps, batchSize, inputSize]});
      const weight = builder.constant(
          {
            type: 'float32',
            dimensions: [numDirections, 3 * hiddenSize, inputSize],
          },
          new Float32Array(numDirections * 3 * hiddenSize * inputSize)
              .fill(0.1));
      const recurrentWeight = builder.constant(
          {
            type: 'float32',
            dimensions: [numDirections, 3 * hiddenSize, hiddenSize],
          },
          new Float32Array(numDirections * 3 * hiddenSize * hiddenSize)
              .fill(0.1));
      const initialHiddenState = builder.constant(
          {type: 'float32', dimensions: [numDirections, batchSize, hiddenSize]},
          new Float32Array(numDirections * batchSize * hiddenSize).fill(0));
      const bias = builder.constant(
          {type: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
          new Float32Array(numDirections * 3 * hiddenSize).fill(0.1));
      const recurrentBias = builder.constant(
          {type: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
          new Float32Array(numDirections * 3 * hiddenSize).fill(0));
      const operands = builder.gru(
          input, weight, recurrentWeight, steps, hiddenSize,
          {bias, recurrentBias, initialHiddenState});
      const graph = builder.build({output: operands[0]});
      const inputs = {
        'input': new Float32Array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
      };
      const outputs = {
        output: new Float32Array(
            utils.sizeOfShape([numDirections, batchSize, hiddenSize])),
      };
      graph.compute(inputs, outputs);
      const expected = [
        0.22391089,
        0.22391089,
        0.22391089,
        0.22391089,
        0.22391089,
        0.1653014,
        0.1653014,
        0.1653014,
        0.1653014,
        0.1653014,
        0.0797327,
        0.0797327,
        0.0797327,
        0.0797327,
        0.0797327,
      ];
      utils.checkValue(outputs.output, expected);

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
});
