'use strict';

const expect = chai.expect;
const assert = chai.assert;

import * as utils from '../utils.js';

describe('test MLContext', () => {
  it('navigator.ml should be a ML', function CheckML() {
    if (typeof window !== 'undefined') {
      expect(navigator.ml).to.be.an.instanceof(ML);
    } else {
      // This case does not need to be tested on Node.js
      this.skip();
    }
  });

  it('ml.createContext should be a function', () => {
    expect(navigator.ml.createContext).to.be.a('function');
  });

  it('ml.createContext should return an Object', async () => {
    expect(await navigator.ml.createContext()).to.be.an.instanceof(Object);
  });

  it('ml.createContext should support MLContextOptions', async () => {
    expect(await navigator.ml.createContext({})).to.be.an.instanceof(Object);
    expect(await navigator.ml.createContext({
      deviceType: 'cpu',
    })).to.be.an.instanceof(Object);
    expect(await navigator.ml.createContext({
      deviceType: 'gpu',
    })).to.be.an.instanceof(Object);
    expect(await navigator.ml.createContext({
      powerPreference: 'default',
    })).to.be.an.instanceof(Object);
    expect(await navigator.ml.createContext({
      powerPreference: 'low-power',
    })).to.be.an.instanceof(Object);
    expect(await navigator.ml.createContext({
      powerPreference: 'high-performance',
    })).to.be.an.instanceof(Object);
    expect(await navigator.ml.createContext({
      deviceType: 'cpu',
      powerPreference: 'default',
    })).to.be.an.instanceof(Object);
    expect(await navigator.ml.createContext({
      deviceType: 'gpu',
      powerPreference: 'default',
    })).to.be.an.instanceof(Object);
    expect(await navigator.ml.createContext({
      deviceType: 'cpu',
      powerPreference: 'high-performance',
    })).to.be.an.instanceof(Object);
    expect(await navigator.ml.createContext({
      deviceType: 'gpu',
      powerPreference: 'low-power',
    })).to.be.an.instanceof(Object);
  });

  it('ml.createContext should throw for invalid options', async () => {
    await expect(navigator.ml.createContext('invalid')).to.be
        .rejectedWith('Invalid options.');
    await expect(navigator.ml.createContext({
      deviceType: 'invalid',
    })).to.be.rejectedWith('Invalid device type.');
    await expect(navigator.ml.createContext({
      powerPreference: 'invalid',
    })).to.be.rejectedWith('Invalid power preference.');
  });
});

describe('test MLContext.compute', () => {
  const desc = {dataType: 'float32', dimensions: [2, 2]};
  const bufferA = new Float32Array(4).fill(1);
  const bufferB = new Float32Array(4).fill(1);
  const bufferC = new Float32Array(4);
  const bufferE = new Float32Array(4);
  const expectedC = [2, 2, 2, 2];
  const expectedE = [3, 3, 3, 3];

  let a; let b; let c; let d; let e; let x; let y; let z;
  let context;
  let builder;
  before(async () => {
    context = await navigator.ml.createContext();
    builder = new MLGraphBuilder(context);
    a = builder.input('a', desc);
    b = builder.input('b', desc);
    c = builder.matmul(a, b);
    d = builder.constant(
        {dataType: 'float32', dimensions: [2, 2]}, new Float32Array(4).fill(1));
    e = builder.add(c, d);
    x = builder.input('x', {dataType: 'float32', dimensions: [3, 2]});
    y = builder.input('y', {dataType: 'float32', dimensions: [2, 4]});
    z = builder.matmul(x, y);
  });

  it('MLContext should have compute method', () => {
    expect(context.compute).to.be.a('function');
  });

  it('MLContext.compute should accept graph, inputs and outputs', async () => {
    const graph = await builder.build({c});
    const inputs = {a: bufferA, b: bufferB};
    const outputs = {c: bufferC};
    const result = await context.compute(graph, inputs, outputs);
    utils.checkValue(result.outputs.c, expectedC);
  });

  it('MLContext.compute should support multiple outputs', async () => {
    const graph = await builder.build({c, e});
    const inputs = {a: bufferA, b: bufferB};
    const outputs = {c: bufferC, e: bufferE};
    const result = await context.compute(graph, inputs, outputs);
    utils.checkValue(result.outputs.c, expectedC);
    utils.checkValue(result.outputs.e, expectedE);
  });

  it('MLContext.compute should support specified outputs', async () => {
    const graph = await builder.build({c, e});
    const inputs = {a: bufferA, b: bufferB};
    let outputs = {c: bufferC};
    const result = await context.compute(graph, inputs, outputs);
    utils.checkValue(result.outputs.c, expectedC);
    expect(result.outputs).not.to.have.property('e');
    outputs = {e: bufferE};
    const resultE = await context.compute(graph, inputs, outputs);
    utils.checkValue(resultE.outputs.e, expectedE);
    expect(resultE.outputs).not.to.have.property('c');
  });

  it('MLContext.compute should support inputs with specified shape',
      async () => {
        const graph = await builder.build({z});
        const shapeX = [3, 2];
        const bufferX = new Float32Array(utils.sizeOfShape(shapeX)).fill(1);
        const shapeY = [2, 4];
        const bufferY = new Float32Array(utils.sizeOfShape(shapeY)).fill(1);
        const inputs = {
          x: bufferX,
          y: bufferY,
        };
        const shapeZ = [shapeX[0], shapeY[1]];
        const outputs = {z: new Float32Array(utils.sizeOfShape(shapeZ))};
        const result = await context.compute(graph, inputs, outputs);
        const expectedZ = new Array(utils.sizeOfShape(shapeZ)).fill(2);
        utils.checkValue(result.outputs.z, expectedZ);
      });

  it('MLContext.compute should throw for non parameter', async () => {
    try {
      await context.compute();
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.compute should throw for non graph', async () => {
    try {
      await context.compute({a: bufferA}, {c: bufferC});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.compute should throw for empty inputs', async () => {
    try {
      const graph = await builder.build({c});
      await context.compute(graph, {}, {c: bufferC});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.compute should throw for non outputs', async () => {
    try {
      const graph = await builder.build({c});
      await context.compute(graph, {a: bufferA, b: bufferB});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.compute should throw for invalid input name', async () => {
    try {
      await context.compute({x: bufferA}, {c: bufferC});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.compute should throw for missing input', async () => {
    try {
      const graph = await builder.build({c});
      await context.compute(graph, {a: bufferA}, {c: bufferC});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.compute should throw for no input resource', async () => {
    try {
      const graph = await builder.build({c});
      await context.compute(graph, {a: {}, b: {}}, {c: bufferC});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.compute should throw for invalid input data', async () => {
    try {
      const graph = await builder.build({c});
      await context.compute(graph, {a: 1, b: 2});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.compute should throw for invalid input dimensions',
      async () => {
        try {
          const graph = await builder.build({c});
          await context.compute(
              graph,
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

  it('MLContext.compute should throw for invalid output name', async () => {
    const graph = await builder.build({c});
    try {
      const inputs = {a: bufferA, b: bufferB};
      const bufferC = new Float32Array(4);
      const outputs = {z: bufferC};
      await context.compute(graph, inputs, outputs);
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.compute should throw for invalid output data', async () => {
    const graph = await builder.build({c});
    try {
      const inputs = {a: bufferA, b: bufferB};
      const outputs = {c: []};
      await context.compute(graph, inputs, outputs);
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.compute should throw for invalid output data length',
      async () => {
        const graph = await builder.build({c});
        try {
          const inputs = {a: bufferA, b: bufferB};
          const bufferC = new Float32Array(1);
          const outputs = {c: bufferC};
          await context.compute(graph, inputs, outputs);
          assert.fail();
        } catch (err) {
          assert(!(err instanceof chai.AssertionError), 'No throwing');
          expect(err).to.be.an.instanceof(Error);
        }
      });

  it('MLGraph should be immutable after creation', async () => {
    const builder = new MLGraphBuilder(context);
    const desc = {dataType: 'float32', dimensions: [2, 2]};
    const a = builder.input('a', desc);
    const bufferB = new Float32Array(4).fill(1);
    let b =
        builder.constant({dataType: 'float32', dimensions: [2, 2]}, bufferB);
    const c = builder.matmul(a, b);
    const bufferA = new Float32Array(4).fill(1);
    const expectedC = [2, 2, 2, 2];
    const graph = await builder.build({c});
    let inputs = {a: bufferA};
    const outputs = {c: bufferC};
    const result = await context.compute(graph, inputs, outputs);
    utils.checkValue(result.outputs.c, expectedC);

    // Change data of constant b should not impact graph compute.
    bufferB.set(new Array(4).fill(2));
    const resultUpdatedBData = await context.compute(graph, inputs, outputs);
    utils.checkValue(resultUpdatedBData.outputs.c, expectedC);

    // Replace b with a new constant should not impact graph compute.
    b = builder.constant({dataType: 'float32', dimensions: [2, 2]}, bufferB);
    const resultUpdatedB = await context.compute(graph, inputs, outputs);
    utils.checkValue(resultUpdatedB.outputs.c, expectedC);

    // Change opearnd type of b should not impact graph compute.
    b = builder.input('b', desc);
    const resultUpdatedBDesc = await context.compute(graph, inputs, outputs);
    utils.checkValue(resultUpdatedBDesc.outputs.c, expectedC);

    // Create new model with new b.
    const graph2 = await builder.build({'c': builder.matmul(a, b)});
    inputs = {'a': bufferA, 'b': bufferB};
    const resultNewModel = await context.compute(graph2, inputs, outputs);
    utils.checkValue(resultNewModel.outputs.c, [4, 4, 4, 4]);
  });

  it('MLContext should not leak memory', async () => {
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
          {dataType: 'float32', dimensions: [steps, batchSize, inputSize]});
      const weight = builder.constant(
          {
            dataType: 'float32',
            dimensions: [numDirections, 3 * hiddenSize, inputSize],
          },
          new Float32Array(numDirections * 3 * hiddenSize * inputSize)
              .fill(0.1));
      const recurrentWeight = builder.constant(
          {
            dataType: 'float32',
            dimensions: [numDirections, 3 * hiddenSize, hiddenSize],
          },
          new Float32Array(numDirections * 3 * hiddenSize * hiddenSize)
              .fill(0.1));
      const initialHiddenState = builder.constant(
          {dataType: 'float32',
            dimensions: [numDirections, batchSize, hiddenSize]},
          new Float32Array(numDirections * batchSize * hiddenSize).fill(0));
      const bias = builder.constant(
          {dataType: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
          new Float32Array(numDirections * 3 * hiddenSize).fill(0.1));
      const recurrentBias = builder.constant(
          {dataType: 'float32', dimensions: [numDirections, 3 * hiddenSize]},
          new Float32Array(numDirections * 3 * hiddenSize).fill(0));
      const operands = builder.gru(
          input, weight, recurrentWeight, steps, hiddenSize,
          {bias, recurrentBias, initialHiddenState});
      const graph = await builder.build({output: operands[0]});
      const inputs = {
        'input': new Float32Array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
      };
      const outputs = {
        output: new Float32Array(
            utils.sizeOfShape([numDirections, batchSize, hiddenSize])),
      };
      const result = await context.compute(graph, inputs, outputs);
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
      utils.checkValue(result.outputs.output, expected);

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
