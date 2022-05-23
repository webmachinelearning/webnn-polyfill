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

// Refer to onnx/models
//   https://github.com/onnx/models/blob/master/workflow_scripts/ort_test_dir_utils.py#L239
// See details of modelFp32AccuracyCriteria setting:
//   https://github.com/webmachinelearning/webnn-polyfill/issues/55
export const modelFp32AccuracyCriteria = new AccuracyCriterion(1e-3, 1e-3);

export function almostEqual(a, b, criteria) {
  const delta = Math.abs(a - b);
  if (delta <= criteria.atol + criteria.rtol * Math.abs(b)) {
    return true;
  } else {
    console.warn(`a(${a}) b(${b}) delta(${delta})`);
    return false;
  }
}

/**
 * Get bitwise of the given value.
 * @param {number} value
 * @param {string} dataType A data type string, like "float32", "int8",
 *     more data type strings, please see:
 *     https://webmachinelearning.github.io/webnn/#enumdef-mloperandtype
 * @return {number} A 64-bit signed integer.
 */
export function getBitwise(value, dataType) {
  const buffer = new ArrayBuffer(8);
  const int64Array = new BigInt64Array(buffer);
  int64Array[0] = value < 0 ? ~BigInt(0) : BigInt(0);
  let typedArray;
  if (dataType === 'float32') {
    typedArray = new Float32Array(buffer);
  } else {
    throw Error(`Data type ${dataType} is not supported.`);
  }
  typedArray[0] = value;
  return int64Array[0];
}

/**
 * Compare the distance between a and b with given ULP distance.
 * @param {number} a
 * @param {number} b
 * @param {number} nulp A BigInt value.
 * @param {string} dataType A data type string, default "float32",
 *     more data type strings, please see:
 *     https://webmachinelearning.github.io/webnn/#enumdef-mloperandtype
 * @return {Boolean} A boolean value:
 *     true: The distance between a and b is greater than given ULP distance.
 *     false: The distance between a and b is less than or equal to given ULP
 *            distance.
 */
export function compareUlp(a, b, nulp = 1n, dataType = 'float32') {
  const aBitwise = getBitwise(a, dataType);
  const bBitwise = getBitwise(b, dataType);
  let distance = aBitwise - bBitwise;
  distance = distance >= 0 ? distance : -distance;
  return distance > nulp;
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
      (accumulator, currentValue) => accumulator * currentValue, 1);
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
  if (!backend) {
    // Use cpu by default for accuracy reason
    // See more details at:
    // https://github.com/webmachinelearning/webnn-polyfill/pull/32#issuecomment-763825323
    backend = 'cpu';
  }
  const tf = navigator.ml.createContext().tf;
  if (tf) {
    const backends = ['webgl', 'webgpu', 'cpu', 'wasm'];
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

export function createActivation(
    builder, activation, input = undefined, options = {}) {
  if (activation === 'relu') {
    return input === undefined ? builder.relu() : builder.relu(input);
  } else if (activation === 'relu6') {
    const clampOptions = {minValue: 0, maxValue: 6};
    return input === undefined ? builder.clamp(clampOptions) :
                                 builder.clamp(input, clampOptions);
  } else if (activation === 'sigmoid') {
    return input === undefined ? builder.sigmoid() : builder.sigmoid(input);
  } else if (activation === 'leakyRelu') {
    return input === undefined ? builder.leakyRelu(options) :
                                 builder.leakyRelu(input, options);
  } else {
    assert(false, `activation ${activation} is not supported`);
  }
}
