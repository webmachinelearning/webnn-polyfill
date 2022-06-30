'use strict';

const expect = chai.expect;
const assert = chai.assert;

import * as utils from '../utils.js';

describe('test MLContext', function() {
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

  it('ml.createContextSync should be a function', () => {
    expect(navigator.ml.createContextSync).to.be.a('function');
  });

  it('ml.createContext should return an Object', async () => {
    expect(await navigator.ml.createContext()).to.be.an.instanceof(Object);
  });

  it('ml.createContextSync should return a MLContext', () => {
    expect(navigator.ml.createContextSync()).to.be.an.instanceof(MLContext);
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

  it('ml.createContextSync should support MLContextOptions', () => {
    expect(navigator.ml.createContextSync({})).to.be.an.instanceof(MLContext);
    expect(navigator.ml.createContextSync({
      deviceType: 'cpu',
    })).to.be.an.instanceof(MLContext);
    expect(navigator.ml.createContextSync({
      deviceType: 'gpu',
    })).to.be.an.instanceof(MLContext);
    expect(navigator.ml.createContextSync({
      powerPreference: 'default',
    })).to.be.an.instanceof(MLContext);
    expect(navigator.ml.createContextSync({
      powerPreference: 'low-power',
    })).to.be.an.instanceof(MLContext);
    expect(navigator.ml.createContextSync({
      powerPreference: 'high-performance',
    })).to.be.an.instanceof(MLContext);
    expect(navigator.ml.createContextSync({
      deviceType: 'cpu',
      powerPreference: 'default',
    })).to.be.an.instanceof(MLContext);
    expect(navigator.ml.createContextSync({
      deviceType: 'gpu',
      powerPreference: 'default',
    })).to.be.an.instanceof(MLContext);
    expect(navigator.ml.createContextSync({
      deviceType: 'cpu',
      powerPreference: 'high-performance',
    })).to.be.an.instanceof(MLContext);
    expect(navigator.ml.createContextSync({
      deviceType: 'gpu',
      powerPreference: 'low-power',
    })).to.be.an.instanceof(MLContext);
  });

  it('ml.createContext should throw for invalid options', async () => {
    try {
      await navigator.ml.createContext('invalid');
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('ml.createContext should throw for invalid deviceType', async () => {
    try {
      await navigator.ml.createContext({
        deviceType: 'invalid',
      });
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('ml.createContext should throw for invalid power preference', async () => {
    try {
      await navigator.ml.createContext({
        powerPreference: 'invalid',
      });
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('ml.createContextSync should throw for invalid options', () => {
    expect(() => navigator.ml.createContextSync('invalid')).to.throw(Error);
  });

  it('ml.createContextSync should throw for invalid deviceType', () => {
    expect(() => navigator.ml.createContextSync({
      deviceType: 'invalid',
    })).to.throw(Error);
  });

  it('ml.createContextSync should throw for invalid power preference', () => {
    expect(() => navigator.ml.createContextSync({
      powerPreference: 'invalid',
    })).to.throw(Error);
  });
  const context = navigator.ml.createContextSync();
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


  it('MLContext should have computeSync method', () => {
    expect(context.compute).to.be.a('function');
  });

  it('MLContext.computeSync should accept graph, inputs and outputs', () => {
    const graph = builder.buildSync({c});
    const inputs = {a: bufferA, b: bufferB};
    const outputs = {c: bufferC};
    context.computeSync(graph, inputs, outputs);
    utils.checkValue(outputs.c, expectedC);
  });

  it('MLContext.computeSync should support multiple outputs', () => {
    const graph = builder.buildSync({c, e});
    const inputs = {a: bufferA, b: bufferB};
    const outputs = {c: bufferC, e: bufferE};
    context.computeSync(graph, inputs, outputs);
    utils.checkValue(outputs.c, expectedC);
    utils.checkValue(outputs.e, expectedE);
  });

  it('MLContext.computeSync should support specified outputs', () => {
    const graph = builder.buildSync({c, e});
    const inputs = {a: bufferA, b: bufferB};
    let outputs = {c: bufferC};
    context.computeSync(graph, inputs, outputs);
    utils.checkValue(outputs.c, expectedC);
    expect(outputs).not.to.have.property('e');
    outputs = {e: bufferE};
    context.computeSync(graph, inputs, outputs);
    utils.checkValue(outputs.e, expectedE);
    expect(outputs).not.to.have.property('c');
  });

  const descX = {type: 'float32', dimensions: [-1, 2]};
  const descY = {type: 'float32', dimensions: [2, -1]};
  const x = builder.input('x', descX);
  const y = builder.input('y', descY);
  const z = builder.matmul(x, y);
  it('MLContext.computeSync should support inputs with specified shape', () => {
    const graph = builder.buildSync({z});
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
    context.computeSync(graph, inputs, outputs);
    const expectedZ = new Array(utils.sizeOfShape(shapeZ)).fill(2);
    utils.checkValue(outputs.z, expectedZ);
  });

  it('MLContext.computeSync should throw for non inputs', () => {
    try {
      context.computeSync();
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.computeSync should throw for empty inputs', () => {
    try {
      context.computeSync({}, {c: bufferC});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.computeSync should throw for non outputs', () => {
    try {
      context.computeSync({a: bufferA, b: bufferB});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.computeSync should throw for invalid input name', () => {
    try {
      context.computeSync({x: bufferA}, {c: bufferC});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.computeSync should throw for missing input', () => {
    try {
      context.computeSync({a: bufferA}, {c: bufferC});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.computeSync should throw for no input resource', () => {
    try {
      context.computeSync({a: {}, b: {}}, {c: bufferC});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.computeSync should throw for invalid input data', () => {
    try {
      context.computeSync({a: 1, b: 2});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.computeSync should throw for invalid input dimensions', () => {
    try {
      context.computeSync(
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

  it('MLContext.computeSync should throw for no dimensions for dynamic shape',
      () => {
        const graph = builder.buildSync({z});
        const shapeX = [3, 2];
        const bufferX = new Float32Array(utils.sizeOfShape(shapeX)).fill(1);
        const shapeY = [2, 4];
        const bufferY = new Float32Array(utils.sizeOfShape(shapeY)).fill(1);
        const inputs = {x: {resource: bufferX}, y: {resource: bufferY}};
        const shapeZ = [shapeX[0], shapeY[1]];
        const outputs = {z: new Float32Array(utils.sizeOfShape(shapeZ))};
        try {
          context.computeSync(graph, inputs, outputs);
          assert.fail();
        } catch (err) {
          assert(!(err instanceof chai.AssertionError), 'No throwing');
          expect(err).to.be.an.instanceof(Error);
        }
      });

  it('MLContext.computeSync should throw for invalid output name', () => {
    const graph = builder.buildSync({c});
    try {
      const inputs = {a: bufferA, b: bufferB};
      const bufferC = new Float32Array(4);
      const outputs = {z: bufferC};
      context.computeSync(graph, inputs, outputs);
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.computeSync should throw for invalid output data', () => {
    const graph = builder.buildSync({c});
    try {
      const inputs = {a: bufferA, b: bufferB};
      const outputs = {c: []};
      context.computeSync(graph, inputs, outputs);
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.computeSync should throw for invalid output data length',
      () => {
        const graph = builder.buildSync({c});
        try {
          const inputs = {a: bufferA, b: bufferB};
          const bufferC = new Float32Array(1);
          const outputs = {c: bufferC};
          context.computeSync(graph, inputs, outputs);
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
    const graph = builder.buildSync({c});
    let inputs = {a: bufferA};
    const outputs = {c: bufferC};
    context.computeSync(graph, inputs, outputs);
    utils.checkValue(outputs.c, expectedC);

    // Change data of constant b should not impact graph compute.
    bufferB.set(new Array(4).fill(2));
    context.computeSync(graph, inputs, outputs);
    utils.checkValue(outputs.c, expectedC);

    // Replace b with a new constant should not impact graph compute.
    b = builder.constant({type: 'float32', dimensions: [2, 2]}, bufferB);
    context.computeSync(graph, inputs, outputs);
    utils.checkValue(outputs.c, expectedC);

    // Change opearnd type of b should not impact graph compute.
    b = builder.input('b', desc);
    context.computeSync(graph, inputs, outputs);
    utils.checkValue(outputs.c, expectedC);

    // Create new model with new b.
    const graph2 = builder.buildSync({'c': builder.matmul(a, b)});
    inputs = {'a': bufferA, 'b': bufferB};
    context.computeSync(graph2, inputs, outputs);
    utils.checkValue(outputs.c, [4, 4, 4, 4]);
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
      const graph = builder.buildSync({output: operands[0]});
      const inputs = {
        'input': new Float32Array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
      };
      const outputs = {
        output: new Float32Array(
            utils.sizeOfShape([numDirections, batchSize, hiddenSize])),
      };
      context.computeSync(graph, inputs, outputs);
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

describe('test MLContext.compute', function() {
  const desc = {type: 'float32', dimensions: [2, 2]};
  const bufferA = new Float32Array(4).fill(1);
  const bufferB = new Float32Array(4).fill(1);
  const bufferC = new Float32Array(4);
  const bufferE = new Float32Array(4);
  const expectedC = [2, 2, 2, 2];
  const expectedE = [3, 3, 3, 3];
  const descX = {type: 'float32', dimensions: [-1, 2]};
  const descY = {type: 'float32', dimensions: [2, -1]};

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
        {type: 'float32', dimensions: [2, 2]}, new Float32Array(4).fill(1));
    e = builder.add(c, d);
    x = builder.input('x', descX);
    y = builder.input('y', descY);
    z = builder.matmul(x, y);
  });

  it('MLContext should have compute method', () => {
    expect(context.compute).to.be.a('function');
  });

  it('MLContext.compute should accept graph, inputs and outputs', async () => {
    const graph = await builder.build({c});
    const inputs = {a: bufferA, b: bufferB};
    const outputs = {c: bufferC};
    await context.compute(graph, inputs, outputs);
    utils.checkValue(outputs.c, expectedC);
  });

  it('MLContext.compute should support multiple outputs', async () => {
    const graph = await builder.build({c, e});
    const inputs = {a: bufferA, b: bufferB};
    const outputs = {c: bufferC, e: bufferE};
    await context.compute(graph, inputs, outputs);
    utils.checkValue(outputs.c, expectedC);
    utils.checkValue(outputs.e, expectedE);
  });

  it('MLContext.compute should support specified outputs', async () => {
    const graph = await builder.build({c, e});
    const inputs = {a: bufferA, b: bufferB};
    let outputs = {c: bufferC};
    await context.compute(graph, inputs, outputs);
    utils.checkValue(outputs.c, expectedC);
    expect(outputs).not.to.have.property('e');
    outputs = {e: bufferE};
    await context.compute(graph, inputs, outputs);
    utils.checkValue(outputs.e, expectedE);
    expect(outputs).not.to.have.property('c');
  });

  it('MLContext.compute should support inputs with specified shape',
      async () => {
        const graph = await builder.build({z});
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
        await context.compute(graph, inputs, outputs);
        const expectedZ = new Array(utils.sizeOfShape(shapeZ)).fill(2);
        utils.checkValue(outputs.z, expectedZ);
      });

  it('MLContext.compute should throw for non inputs', async () => {
    try {
      await context.compute();
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.compute should throw for empty inputs', async () => {
    try {
      await context.compute({}, {c: bufferC});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.compute should throw for non outputs', async () => {
    try {
      await context.compute({a: bufferA, b: bufferB});
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
      await context.compute({a: bufferA}, {c: bufferC});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.compute should throw for no input resource', async () => {
    try {
      await context.compute({a: {}, b: {}}, {c: bufferC});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.compute should throw for invalid input data', async () => {
    try {
      await context.compute({a: 1, b: 2});
      assert.fail();
    } catch (err) {
      assert(!(err instanceof chai.AssertionError), 'No throwing');
      expect(err).to.be.an.instanceof(Error);
    }
  });

  it('MLContext.compute should throw for invalid input dimensions',
      async () => {
        try {
          await context.compute(
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

  it('MLContext.compute should throw for no dimensions for dynamic shape',
      async () => {
        const graph = await builder.build({z});
        const shapeX = [3, 2];
        const bufferX = new Float32Array(utils.sizeOfShape(shapeX)).fill(1);
        const shapeY = [2, 4];
        const bufferY = new Float32Array(utils.sizeOfShape(shapeY)).fill(1);
        const inputs = {x: {resource: bufferX}, y: {resource: bufferY}};
        const shapeZ = [shapeX[0], shapeY[1]];
        const outputs = {z: new Float32Array(utils.sizeOfShape(shapeZ))};
        try {
          await context.compute(graph, inputs, outputs);
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
    const desc = {type: 'float32', dimensions: [2, 2]};
    const a = builder.input('a', desc);
    const bufferB = new Float32Array(4).fill(1);
    let b = builder.constant({type: 'float32', dimensions: [2, 2]}, bufferB);
    const c = builder.matmul(a, b);
    const bufferA = new Float32Array(4).fill(1);
    const expectedC = [2, 2, 2, 2];
    const graph = await builder.build({c});
    let inputs = {a: bufferA};
    const outputs = {c: bufferC};
    await context.compute(graph, inputs, outputs);
    utils.checkValue(outputs.c, expectedC);

    // Change data of constant b should not impact graph compute.
    bufferB.set(new Array(4).fill(2));
    await context.compute(graph, inputs, outputs);
    utils.checkValue(outputs.c, expectedC);

    // Replace b with a new constant should not impact graph compute.
    b = builder.constant({type: 'float32', dimensions: [2, 2]}, bufferB);
    await context.compute(graph, inputs, outputs);
    utils.checkValue(outputs.c, expectedC);

    // Change opearnd type of b should not impact graph compute.
    b = builder.input('b', desc);
    await context.compute(graph, inputs, outputs);
    utils.checkValue(outputs.c, expectedC);

    // Create new model with new b.
    const graph2 = await builder.build({'c': builder.matmul(a, b)});
    inputs = {'a': bufferA, 'b': bufferB};
    await context.compute(graph2, inputs, outputs);
    utils.checkValue(outputs.c, [4, 4, 4, 4]);
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
        'input': new Float32Array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
      };
      const outputs = {
        output: new Float32Array(
            utils.sizeOfShape([numDirections, batchSize, hiddenSize])),
      };
      await context.compute(graph, inputs, outputs);
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
