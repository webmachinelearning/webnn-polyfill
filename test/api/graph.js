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
  const expectedC = [2, 2, 2, 2];
  const expectedE = [3, 3, 3, 3];

  it('MLGraph should have compute method', async () => {
    const graph = await builder.build({c});
    expect(graph.compute).to.be.a('function');
  });

  it('MLGraph.compute should accept inputs and return outputs', async () => {
    const graph = await builder.build({c});
    const inputs = {a: {data: bufferA}, b: {data: bufferB}};
    const outputs = await graph.compute(inputs);
    expect(outputs).to.be.a('object');
    expect(outputs).to.have.property('c');
    expect(outputs.c).to.have.property('data');
    utils.checkValue(outputs.c.data, expectedC);
    expect(outputs.c).to.have.property('dimensions');
    utils.checkShape(outputs.c.dimensions, [2, 2]);
  });

  it('MLGraph.compute should accept inputs and pre-allocated outputs',
     async () => {
       const graph = await builder.build({c});
       const inputs = {a: {data: bufferA}, b: {data: bufferB}};
       const bufferC = new Float32Array(4);
       const outputs = {c: {data: bufferC}};
       await graph.compute(inputs, outputs);
       utils.checkValue(bufferC, expectedC);
     });

  it('MLGraph.compute should support multiple outputs', async () => {
    const graph = await builder.build({c, e});
    const inputs = {a: {data: bufferA}, b: {data: bufferB}};
    const outputs = await graph.compute(inputs);
    expect(outputs).to.be.a('object');
    expect(outputs).to.have.property('c');
    expect(outputs.c).to.have.property('data');
    utils.checkValue(outputs.c.data, expectedC);
    expect(outputs.c).to.have.property('dimensions');
    utils.checkShape(outputs.c.dimensions, [2, 2]);
    expect(outputs).to.have.property('e');
    expect(outputs.e).to.have.property('data');
    utils.checkValue(outputs.e.data, expectedE);
    expect(outputs.e).to.have.property('dimensions');
    utils.checkShape(outputs.e.dimensions, [2, 2]);
  });

  it('MLGraph.compute should support specified outputs', async () => {
    const graph = await builder.build({c, e});
    const inputs = {a: {data: bufferA}, b: {data: bufferB}};
    let outputs = await graph.compute(inputs, {c});
    expect(outputs).to.be.a('object');
    expect(outputs).to.have.property('c');
    expect(outputs.c).to.have.property('data');
    utils.checkValue(outputs.c.data, expectedC);
    expect(outputs.c).to.have.property('dimensions');
    utils.checkShape(outputs.c.dimensions, [2, 2]);
    expect(outputs).not.to.have.property('e');
    outputs = await graph.compute(inputs, {e});
    expect(outputs).to.be.a('object');
    expect(outputs).to.have.property('e');
    expect(outputs.e).to.have.property('data');
    utils.checkValue(outputs.e.data, expectedE);
    expect(outputs.e).to.have.property('dimensions');
    utils.checkShape(outputs.e.dimensions, [2, 2]);
    expect(outputs).not.to.have.property('c');
  });

  const descX = {type: 'float32', dimensions: [-1, 2]};
  const descY = {type: 'float32', dimensions: [2, -1]};
  const x = builder.input('x', descX);
  const y = builder.input('y', descY);
  const z = builder.matmul(x, y);
  it('MLGraph.compute should support inputs with specified shape', async () => {
    const graph = await builder.build({z});
    const shapeX = [3, 2];
    const bufferX = new Float32Array(utils.sizeOfShape(shapeX)).fill(1);
    const shapeY = [2, 4];
    const bufferY = new Float32Array(utils.sizeOfShape(shapeY)).fill(1);
    const inputs = {
      x: {data: bufferX, dimensions: shapeX},
      y: {data: bufferY, dimensions: shapeY},
    };
    const outputs = await graph.compute(inputs);
    expect(outputs).to.be.a('object');
    expect(outputs).to.have.property('z');
    expect(outputs.z).to.have.property('data');
    const shapeZ = [shapeX[0], shapeY[1]];
    const expectedZ = new Array(utils.sizeOfShape(shapeZ)).fill(2);
    utils.checkValue(outputs.z.data, expectedZ);
    expect(outputs.z).to.have.property('dimensions');
    utils.checkShape(outputs.z.dimensions, shapeZ);
  });

  it('MLGraph.compute should throw for non inputs', async () => {
    const graph = await builder.build({c});
    try {
      await graph.compute();
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLGraph.compute should throw for empty inputs', async () => {
    const graph = await builder.build({c});
    try {
      await graph.compute({});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLGraph.compute should throw for invalid input name', async () => {
    const graph = await builder.build({c});
    try {
      await graph.compute({x: {data: bufferA}});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLGraph.compute should throw for missing input', async () => {
    const graph = await builder.build({c});
    try {
      await graph.compute({a: {data: bufferA}});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLGraph.compute should throw for no input data', async () => {
    const graph = await builder.build({c});
    try {
      await graph.compute({a: {}, b: {}});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLGraph.compute should throw for invalid input data', async () => {
    const graph = await builder.build({c});
    try {
      await graph.compute({a: {data: 1}, b: {data: 2}});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLGraph.compute should throw for invalid input dimensions', async () => {
    const graph = await builder.build({c});
    try {
      await graph.compute(
          {a: {data: bufferA, dimensions: [2]}, b: {data: bufferB}});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLGraph.compute should throw for no dimensions for dynamic shape',
     async () => {
       const graph = await builder.build({z});
       const shapeX = [3, 2];
       const bufferX = new Float32Array(utils.sizeOfShape(shapeX)).fill(1);
       const shapeY = [2, 4];
       const bufferY = new Float32Array(utils.sizeOfShape(shapeY)).fill(1);
       const inputs = {x: {data: bufferX}, y: {data: bufferY}};
       try {
         await graph.compute(inputs);
         assert.fail();
       } catch (err) {
         assert(!(err instanceof chai.AssertionError), 'No throwing');
         expect(err).to.be.an.instanceof(Error);
       }
     });

  it('MLGraph.compute should throw for invalid output name', async () => {
    const graph = await builder.build({c});
    try {
      const inputs = {a: {data: bufferA}, b: {data: bufferB}};
      const bufferC = new Float32Array(4);
      const outputs = {z: {data: bufferC}};
      await graph.compute(inputs, outputs);
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLGraph.compute should throw for invalid output data', async () => {
    const graph = await builder.build({c});
    try {
      const inputs = {a: {data: bufferA}, b: {data: bufferB}};
      const outputs = {c: {data: []}};
      await graph.compute(inputs, outputs);
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLGraph.compute should throw for invalid output data length',
     async () => {
       const graph = await builder.build({c});
       try {
         const inputs = {a: {data: bufferA}, b: {data: bufferB}};
         const bufferC = new Float32Array(1);
         const outputs = {c: {data: bufferC}};
         await graph.compute(inputs, outputs);
         assert.fail();
       } catch (err) {
         assert(!(err instanceof chai.AssertionError), 'No throwing');
         expect(err).to.be.an.instanceof(Error);
       }
     });

  it('MLGraph should be immutable after creation', async () => {
    const builder = new MLGraphBuilder(context);
    const desc = {type: 'float32', dimensions: [2, 2]};
    const a = builder.input('a', desc);
    const bufferB = new Float32Array(4).fill(1);
    let b = builder.constant({type: 'float32', dimensions: [2, 2]}, bufferB);
    const c = builder.matmul(a, b);
    const bufferA = new Float32Array(4).fill(1);
    const expectedC = [2, 2, 2, 2];
    const graph = await builder.build({c});
    let inputs = {a: {data: bufferA}};
    let outputs = await graph.compute(inputs);
    utils.checkValue(outputs.c.data, expectedC);
    utils.checkShape(outputs.c.dimensions, [2, 2]);

    // Change data of constant b should not impact model execution.
    bufferB.set(new Array(4).fill(2));
    outputs = await graph.compute(inputs);
    utils.checkValue(outputs.c.data, expectedC);
    utils.checkShape(outputs.c.dimensions, [2, 2]);

    // Replace b with a new constant should not impact model execution.
    b = builder.constant({type: 'float32', dimensions: [2, 2]}, bufferB);
    outputs = await graph.compute(inputs);
    utils.checkValue(outputs.c.data, expectedC);
    utils.checkShape(outputs.c.dimensions, [2, 2]);

    // Change opearnd type of b should not impact model execution.
    b = builder.input('b', desc);
    outputs = await graph.compute(inputs);
    utils.checkValue(outputs.c.data, expectedC);
    utils.checkShape(outputs.c.dimensions, [2, 2]);

    // Create new model with new b.
    const graph2 = await builder.build({'c': builder.matmul(a, b)});
    inputs = {'a': {data: bufferA}, 'b': {data: bufferB}};
    outputs = await graph2.compute(inputs);
    utils.checkValue(outputs.c.data, [4, 4, 4, 4]);
    utils.checkShape(outputs.c.dimensions, [2, 2]);
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
      const graph = await builder.build({output: operands[0]});
      const inputs = {
        'input': {
          data: new Float32Array(
              [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
        },
      };
      const outputs = await graph.compute(inputs);
      utils.checkShape(
          outputs.output.dimensions, [numDirections, batchSize, hiddenSize]);
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
      utils.checkValue(outputs.output.data, expected);

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
