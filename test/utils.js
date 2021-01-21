'use strict';

import {numpy} from './lib/numpy.js';

const assert = chai.assert;

export class AccuracyCriterion {
  constructor(atol, rtol) {
    this.atol = atol;
    this.rtol = rtol;
  }
}

export const opFp32AccuracyCriteria =
    new AccuracyCriterion(1e-6, 5.0 * 1.1920928955078125e-7);

// The following 2 constants were used for converted tests from NNAPI CTS
export const ctsFp32RestrictAccuracyCriteria =
    new AccuracyCriterion(1e-5, 5.0 * 1.1920928955078125e-7);
export const ctsFp32RelaxedAccuracyCriteria =
    new AccuracyCriterion(5.0 * 0.0009765625, 5.0 * 0.0009765625);

export function almostEqual(a, b, criteria) {
  const delta = Math.abs(a - b);
  if (delta <= criteria.atol + criteria.rtol * Math.abs(b)) {
    return true;
  } else {
    console.warn(`a(${a}) b(${b}) delta(${delta})`);
    return false;
  }
}

export function checkValue(
    output, expected, criteria = opFp32AccuracyCriteria) {
  assert.isTrue(output.length === expected.length);
  for (let i = 0; i < output.length; ++i) {
    assert.isTrue(almostEqual(output[i], expected[i], criteria));
  }
}

export function sizeOfShape(array) {
  return array.reduce(
      (accumulator, currentValue) => accumulator * currentValue);
}

export function checkShape(shape, expected) {
  assert.isTrue(shape.length === expected.length);
  for (let i = 0; i < shape.length; ++i) {
    assert.isTrue(shape[i] === expected[i]);
  }
}

async function readFromNpy(fileName) {
  const dataTypeMap = new Map([
    ['f2', {type: 'float16', array: Uint16Array}],
    ['f4', {type: 'float32', array: Float32Array}],
    ['f8', {type: 'float64', array: Float64Array}],
    ['i1', {type: 'int8', array: Int8Array}],
    ['i2', {type: 'int16', array: Int16Array}],
    ['i4', {type: 'int32', array: Int32Array}],
    ['i8', {type: 'int64', array: BigInt64Array}],
    ['u1', {type: 'uint8', array: Uint8Array}],
    ['u2', {type: 'uint16', array: Uint16Array}],
    ['u4', {type: 'uint32', array: Uint32Array}],
    ['u8', {type: 'uint64', array: BigUint64Array}],
  ]);
  let buffer;
  if (typeof fs !== 'undefined') {
    buffer = fs.readFileSync(fileName);
  } else {
    const response = await fetch(fileName);
    buffer = await response.arrayBuffer();
  }
  const npArray = new numpy.Array(new Uint8Array(buffer));
  if (!dataTypeMap.has(npArray.dataType)) {
    throw new Error(`Data type ${npArray.dataType} is not supported.`);
  }
  const dimensions = npArray.shape;
  const type = dataTypeMap.get(npArray.dataType).type;
  const TypedArrayConstructor = dataTypeMap.get(npArray.dataType).array;
  const typedArray = new TypedArrayConstructor(sizeOfShape(dimensions));
  const dataView = new DataView(npArray.data.buffer);
  const littleEndian = npArray.byteOrder === '<';
  for (let i = 0; i < sizeOfShape(dimensions); ++i) {
    typedArray[i] = dataView[`get` + type[0].toUpperCase() + type.substr(1)](
        i * TypedArrayConstructor.BYTES_PER_ELEMENT, littleEndian);
  }
  return {buffer: typedArray, type, dimensions};
}

export async function createTypedArrayFromNpy(fileName) {
  const data = await readFromNpy(fileName);
  return data.buffer;
}

export async function buildConstantFromNpy(builder, fileName) {
  const data = await readFromNpy(fileName);
  return builder.constant(
      {type: data.type, dimensions: data.dimensions}, data.buffer);
}

// Refer to Implicit padding algorithms of Android NNAPI:
// https://developer.android.com/ndk/reference/group/neural-networks#group___neural_networks_1gab72e9e6263fd5b015bb7f41ec18ce220
export function computeExplicitPadding(
    inputSize, stride, filterSize, dilation = 1) {
  const outSize = Math.ceil(inputSize / stride);
  const effectiveFilterSize = (filterSize - 1) * dilation + 1;
  const neededInput = (outSize - 1) * stride + effectiveFilterSize;
  const totalPadding = Math.max(0, neededInput - inputSize);
  const paddingToBeginning = Math.floor(totalPadding / 2);
  const paddingToEnd = Math.floor((totalPadding + 1) / 2);
  return [paddingToBeginning, paddingToEnd];
}

export async function setPolyfillBackend(backend) {
  const tf = navigator.ml.getNeuralNetworkContext().tf;
  if (tf) {
    const backends = ['webgl', 'cpu'];
    if (!backends.includes(backend)) {
      if (backend) {
        console.warn(`webnn-polyfill doesn't support ${backend} backend.`);
      }
    } else {
      if (!(await tf.setBackend(backend))) {
        console.error(`Failed to set tf.js backend ${backend}.`);
      }
    }
    await tf.ready();
    console.info(
        `webnn-polyfill uses tf.js ${tf.version_core}` +
        ` ${tf.getBackend()} backend.`);
  }
}
