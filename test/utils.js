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
 *     https://webmachinelearning.github.io/webnn/#enumdef-mloperanddatatype
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
 *     https://webmachinelearning.github.io/webnn/#enumdef-mloperanddatatype
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

export function checkDataType(dataType, expectedDataType) {
  // console.log(`${dataType.length} -- ${expectedDataType.length}`)
  assert.isTrue(dataType === expectedDataType);
}

export function checkShape(shape, expected) {
  assert.isTrue(shape.length === expected.length);
  for (let i = 0; i < shape.length; ++i) {
    assert.isTrue(shape[i] === expected[i]);
  }
}

async function readFromNpy(fileName) {
  const dataTypeMap = new Map([
    ['f2', {dataType: 'float16', array: Uint16Array}],
    ['f4', {dataType: 'float32', array: Float32Array}],
    ['f8', {dataType: 'float64', array: Float64Array}],
    ['i1', {dataType: 'int8', array: Int8Array}],
    ['i2', {dataType: 'int16', array: Int16Array}],
    ['i4', {dataType: 'int32', array: Int32Array}],
    ['i8', {dataType: 'int64', array: BigInt64Array}],
    ['u1', {dataType: 'uint8', array: Uint8Array}],
    ['u2', {dataType: 'uint16', array: Uint16Array}],
    ['u4', {dataType: 'uint32', array: Uint32Array}],
    ['u8', {dataType: 'uint64', array: BigUint64Array}],
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
  const dataType = dataTypeMap.get(npArray.dataType).dataType;
  const TypedArrayConstructor = dataTypeMap.get(npArray.dataType).array;
  const dataView = new Uint8Array(npArray.data.buffer);
  const dataView2 = dataView.slice();
  const typedArray = new TypedArrayConstructor(dataView2.buffer);
  return {buffer: typedArray, dataType, dimensions};
}

export async function createTypedArrayFromNpy(fileName) {
  const data = await readFromNpy(fileName);
  return data.buffer;
}

export async function buildConstantFromNpy(builder, fileName) {
  const data = await readFromNpy(fileName);
  return builder.constant(
      {dataType: data.dataType, dimensions: data.dimensions}, data.buffer);
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
  const context = await navigator.ml.createContext();
  await readyPolyfillBackend(context, backend);
}

export async function setPolyfillBackendSW(backend) {
  if (!backend) {
    // Use cpu by default for accuracy reason
    // See more details at:
    // https://github.com/webmachinelearning/webnn-polyfill/pull/32#issuecomment-763825323
    backend = 'cpu';
  }
  const worker = new Worker('worker_context.js');
  worker.postMessage('');
  const context = await new Promise((resolve, reject) => {
    // Receive a message from the worker
    worker.onmessage = function(event) {
      resolve(event.data);
    };
  });
  await readyPolyfillBackend(context, backend);
}

async function readyPolyfillBackend(context, backend) {
  const tf = context.tf;
  if (tf) {
    const backends = ['webgl', 'webgpu', 'cpu', 'wasm'];
    if (!backends.includes(backend)) {
      if (backend) {
        console.warn(`webnn-polyfill doesn't support ${backend} backend.`);
      }
    } else {
      if (backend === 'wasm') {
        const wasm = context.wasm;
        wasm.setWasmPaths(`https://unpkg.com/@tensorflow/tfjs-backend-wasm@${tf.version_core}/dist/`);
      }
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

// Derive from
// https://github.com/webmachinelearning/webnn-baseline/blob/main/src/lib/compute-padding.js
/**
 * Compute the beginning and ending pad given input, filter and stride sizes.
 * @param {String} autoPad
 * @param {Number} inputSize
 * @param {Number} effectiveFilterSize
 * @param {Number} stride
 * @param {Number} outputPadding
 * @return {Array} [paddingBegin, paddingEnd]
 */
function computePadding1DForAutoPad(
    autoPad, inputSize, effectiveFilterSize, stride, outputPadding) {
  let totalPadding;
  if (outputPadding === undefined) {
  // for conv2d
    const outSize = Math.ceil(inputSize / stride);
    const neededInput = (outSize - 1) * stride + effectiveFilterSize;
    totalPadding = neededInput > inputSize ? neededInput - inputSize : 0;
  } else {
  // for convTranspose2d
  // totalPadding = beginning padding + ending padding
  // SAME_UPPER or SAME_LOWER mean pad the input so that
  //   output size = input size * strides
  // output size = (input size - 1) * stride + effectiveFilterSize
  //     - beginning padding - ending padding + output padding
    totalPadding = (inputSize - 1) * stride + effectiveFilterSize +
      outputPadding - inputSize * stride;
  }
  let paddingBegin;
  let paddingEnd;
  switch (autoPad) {
    case 'same-upper':
      paddingBegin = Math.floor(totalPadding / 2);
      paddingEnd = Math.floor((totalPadding + 1) / 2);
      break;
    case 'same-lower':
      paddingBegin = Math.floor((totalPadding + 1) / 2);
      paddingEnd = Math.floor(totalPadding / 2);
      break;
    default:
      throw new Error('The autoPad is invalid.');
  }
  return [paddingBegin, paddingEnd];
}

// Compute explicit padding given input sizes, filter sizes, strides, dilations
// and auto pad mode 'same-upper' or 'same-lower'.
export function computePadding2DForAutoPad(
    inputSizes, filterSizes, strides, dilations, autoPad = 'same-upper') {
  const [inputHeight, inputWidth] = inputSizes;
  const [filterHeight, filterWidth] = filterSizes;
  const [strideHeight, strideWidth] = strides ? strides : [1, 1];
  const [dilationHeight, dilationWidth] = dilations ? dilations: [1, 1];
  const effectiveFilterHeight = (filterHeight - 1) * dilationHeight + 1;
  const effectiveFilterWidth = (filterWidth - 1) * dilationWidth + 1;
  const [beginningPaddingHeight, endingPaddingHeight] =
    computePadding1DForAutoPad(
        autoPad, inputHeight, effectiveFilterHeight, strideHeight);
  const [beginningPaddingWidth, endingPaddingWidth] =
    computePadding1DForAutoPad(
        autoPad, inputWidth, effectiveFilterWidth, strideWidth);
  return [beginningPaddingHeight, endingPaddingHeight,
    beginningPaddingWidth, endingPaddingWidth];
}

export function buildMaxPool2d(input, options, builder, layout = 'nhwc') {
  const inputSizes = layout =='nhwc' ? [input.shape()[1], input.shape()[2]] :
      [input.shape()[2], input.shape()[3]];
  options.padding = computePadding2DForAutoPad(
      inputSizes, options.windowDimensions, options.strides, options.dilations);
  return builder.maxPool2d(input, options);
}
