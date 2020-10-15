'use strict';

const expect = chai.expect;

import * as utils from '../utils.js';

describe('test Compilation', function() {
  const nn = navigator.ml.getNeuralNetworkContext();
  const builder = nn.createModelBuilder();
  const desc = {type: 'float32', dimensions: [2, 2]};
  const a = builder.input('a', desc);
  const b = builder.input('b', desc);
  const c = builder.matmul(a, b);
  const model = builder.createModel({c});
  let compiledModel;
  before(async () => {
    compiledModel = await model.compile();
  });
  const bufferA = new Float32Array(4).fill(1);
  const bufferB = new Float32Array(4).fill(1);
  const expected = [2, 2, 2, 2];

  it('Compilation should have compute method', () => {
    expect(compiledModel.compute).to.be.a('function');
  });

  it('Compilation.compute should accept inputs and return outputs',
     async () => {
       const inputs = {a: {buffer: bufferA}, b: {buffer: bufferB}};
       const outputs = await compiledModel.compute(inputs);
       expect(outputs).to.be.a('object');
       expect(outputs).to.have.property('c');
       expect(outputs.c).to.have.property('buffer');
       utils.checkValue(outputs.c.buffer, expected);
       expect(outputs.c).to.have.property('dimensions');
       utils.checkShape(outputs.c.dimensions, [2, 2]);
     });

  it('Compilation.compute should accept inputs and pre-allocated outputs',
     async () => {
       const inputs = {a: {buffer: bufferA}, b: {buffer: bufferB}};
       const bufferC = new Float32Array(4);
       const outputs = {c: {buffer: bufferC}};
       await compiledModel.compute(inputs, outputs);
       utils.checkValue(bufferC, expected);
     });

  it('Compilation.compute should support inputs with specified shape',
     async () => {
       const descA = {type: 'float32', dimensions: [-1, 2]};
       const descB = {type: 'float32', dimensions: [2, -1]};
       const a = builder.input('a', descA);
       const b = builder.input('b', descB);
       const c = builder.matmul(a, b);
       const model = builder.createModel({c});
       const compiledModel = await model.compile();
       const shapeA = [3, 2];
       const bufferA = new Float32Array(utils.sizeOfShape(shapeA)).fill(1);
       const shapeB = [2, 4];
       const bufferB = new Float32Array(utils.sizeOfShape(shapeB)).fill(1);
       const inputs = {
         a: {buffer: bufferA, dimensions: shapeA},
         b: {buffer: bufferB, dimensions: shapeB},
       };
       const outputs = await compiledModel.compute(inputs);
       expect(outputs).to.be.a('object');
       expect(outputs).to.have.property('c');
       expect(outputs.c).to.have.property('buffer');
       const expected = new Array(12).fill(2);
       utils.checkValue(outputs.c.buffer, expected);
       console.log(outputs.c.buffer);
       expect(outputs.c).to.have.property('dimensions');
       utils.checkShape(outputs.c.dimensions, [shapeA[0], shapeB[1]]);
     });
});
