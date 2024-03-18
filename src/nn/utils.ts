import * as tf from '@tensorflow/tfjs-core';
import {ExplicitPadding} from '@tensorflow/tfjs-core/dist/ops/conv_util';

import {MLOperand, MLOperandDescriptor, MLOperandDataType} from './operand';
import {ArrayBufferView as TypedArray} from './types';

export function assert(expr: boolean, msg: string): void {
  if (!expr) {
    throw new Error(msg);
  }
}

export function isBoolean(value: unknown): boolean {
  return typeof value === 'boolean';
}

export function isInteger(value: unknown): boolean {
  return typeof value === 'number' && Number.isInteger(value);
}

export function isUnsignedInteger(value: unknown): boolean {
  return typeof value === 'number' && Number.isInteger(value) && value >= 0;
}

export function isIntegerArray(array: number[]): boolean {
  return array instanceof Array && array.every(v => isInteger(v));
}

export function isUnsignedIntegerArray(array: number[]): boolean {
  return array instanceof Array && array.every(v => isUnsignedInteger(v));
}

export function isPositiveIntegerOrNullArray(array: Array<(number | null)>):
  boolean {
  return array instanceof Array &&
      array.every(v => (isInteger(v) && v > 0) || v === null);
}

export function isTypedArray(array: ArrayBufferView): boolean {
  return array instanceof Float32Array || array instanceof Int32Array ||
      array instanceof Uint32Array || array instanceof Int16Array ||
      array instanceof Uint16Array || array instanceof Int8Array ||
      array instanceof Uint8Array;
}

export function isValidResample2dAxes(array: number[]): boolean {
  if (array[0] === 0 && array[1] === 1) {
    return true;
  } else if (array[0] === 1 && array[1] === 2) {
    return true;
  } else if (array[0] === 2 && array[1] === 3) {
    return true;
  } else {
    return false;
  }
}

export function getTypedArray(type: MLOperandDataType): Float32ArrayConstructor|
    Int32ArrayConstructor|Uint32ArrayConstructor|Uint16ArrayConstructor|
    Int8ArrayConstructor|Uint8ArrayConstructor {
  if (type === 'float32') {
    return Float32Array;
  } else if (type === 'int32') {
    return Int32Array;
  } else if (type === 'uint32') {
    return Uint32Array;
  } else if (type === 'float16') {
    return Uint16Array;
  } else if (type === 'int8') {
    return Int8Array;
  } else if (type === 'uint8') {
    return Uint8Array;
  } else {
    throw new Error('Type is not supported.');
  }
}

export function cloneTypedArray(value: TypedArray): TypedArray {
  let array;
  if (value instanceof Float32Array) {
    array = new Float32Array(value.length);
  } else if (value instanceof Int32Array) {
    array = new Int32Array(value.length);
  } else if (value instanceof Uint32Array) {
    array = new Uint32Array(value.length);
  } else if (value instanceof Uint16Array) {
    array = new Uint16Array(value.length);
  } else if (value instanceof Int8Array) {
    array = new Int8Array(value.length);
  } else if (value instanceof Uint8Array) {
    array = new Uint8Array(value.length);
  } else {
    throw new Error('Type is not supported.');
  }
  array.set(value);
  return array;
}

export function getDataType(type: MLOperandDataType): tf.DataType {
  if (type === 'float32') {
    return 'float32';
  } else if (type === 'int32') {
    return 'int32';
  } else {
    throw new Error('The operand type is not supported by TF.js.');
  }
}

export function createOperandDescriptorFromTensor(tensor: tf.Tensor):
    MLOperandDescriptor {
  let dataType: MLOperandDataType;
  if (tensor.dtype === 'float32') {
    dataType = MLOperandDataType.float32;
  } else if (tensor.dtype === 'int32') {
    dataType = MLOperandDataType.int32;
  }
  return {dataType, dimensions: tensor.shape} as MLOperandDescriptor;
}

export function validateOperandDescriptor(desc: MLOperandDescriptor): void {
  assert(desc.dataType in MLOperandDataType, 'The operand type is invalid.');
  if (desc.dimensions) {
    assert(isIntegerArray(desc.dimensions), 'The dimensions is invalid.');
  }
}

export function isDyanmicShape(dimensions: number[]): boolean {
  return !dimensions.every(x => x > 0);
}

export function validateTypedArray(
    value: TypedArray, type: MLOperandDataType, dimensions: number[]): void {
  assert(isTypedArray(value), 'The value is not a typed array.');
  assert(value instanceof getTypedArray(type), 'The type of value is invalid.');
  assert(
      value.length === sizeFromDimensions(dimensions),
      `the value length ${value.length} is invalid, size of ` +
          `[${dimensions}] ${sizeFromDimensions(dimensions)} ` +
          'is expected.');
}

export function validateValueType(value: number, type: MLOperandDataType):
 void {
  if (type === MLOperandDataType.int32) {
    assert(Number.isInteger(value), 'the value is not an int32.');
  } else if (type === MLOperandDataType.uint32) {
    assert(
        Number.isInteger(value) && value >= 0, 'the value is not an uint32.');
  } else if (type === MLOperandDataType.int8) {
    assert(
        Number.isInteger(value) && value >= -128 && value <= 127,
        'the value is not an int8.');
  } else if (type === MLOperandDataType.uint8) {
    assert(
        Number.isInteger(value) && value >= 0 && value <= 255,
        'the value is not an uint8.');
  }
}

export function createTensor(
    desc: MLOperandDescriptor,
    value: ArrayBufferView|number): tf.Tensor {
  const dtype: tf.DataType = getDataType(desc.dataType);
  if (desc.dimensions !== undefined) {
    assert(
        isTypedArray(value as ArrayBufferView),
        'Only ArrayBufferView value is supported.');
    const array = value as TypedArray;
    validateTypedArray(array, desc.dataType, desc.dimensions);
    const clonedArray = cloneTypedArray(array);
    return tf.tensor(clonedArray, desc.dimensions, dtype);
  } else {
    if (typeof value === 'number') {
      validateValueType(value, desc.dataType);
      return tf.scalar(value, dtype);
    } else {
      validateTypedArray(value as TypedArray, desc.dataType, desc.dimensions);
      return tf.scalar((value as TypedArray)[0], dtype);
    }
  }
}

export function sizeFromDimensions(dim: number[]): number {
  if (dim === undefined || (isIntegerArray(dim) && dim.length === 0)) {
    // scalar
    return 1;
  } else {
    return dim.reduce(
        (accumulator, currentValue) =>
            currentValue > 0 ? accumulator * currentValue : accumulator,
        1);
  }
}

export function validateOperand(input: MLOperand, name = ''): void {
  assert(
      input instanceof MLOperand, `The parameter ${name} is not an operand.`);
}

export function validateOptionalOperand(input: MLOperand, name = ''): void {
  assert(
      input === undefined || input instanceof MLOperand,
      `The parameter ${name} is not an optional operand.`);
}

export function validateAxes(axes: number[], rank: number): boolean {
  if (typeof axes !== 'undefined' && axes.length > 0) {
    for (let i = 0; i < axes.length; ++i) {
      if (axes[i] >= rank || axes[i] < 0) {
        return false;
      }
    }
  }
  return true;
}

export function product(array: number[]): number {
  return array.reduce(
      (accumulator, currentValue) => accumulator * currentValue);
}

export function checkShape(actual: number[], expected: number[]): void {
  assert(actual.length === expected.length,
    `The actual length ${actual.length} is not equal to expected length ` +
    `${expected.length}.`);
  for (let i = 0; i < actual.length; ++i) {
    assert(
      actual[i] === expected[i],
      `${actual[i]} is not equal to ${expected[i]} of index ${i}.`
      );
  }
}

export function getPaddings(
    padding: [number, number, number, number]): ExplicitPadding {
  // input layout: NHWC
  // WebNN padding:
  //   [beginning_height, ending_height, beginning_width, ending_width]
  // tf.conv2d NHWC should be in the following form:
  //   [[0, 0], [pad_top,pad_bottom], [pad_left, pad_right], [0, 0]]
  const resultPadding: ExplicitPadding = [
      [0, 0], [padding[0], padding[1]], [padding[2], padding[3]], [0, 0]
    ] as ExplicitPadding;

  return resultPadding;
}

export function getBroadcastShape(shapeA: number[], shapeB: number[]):
    number[] {
  // According to General Broadcasting Rules on
  //   https://numpy.org/doc/stable/user/basics.broadcasting.html.
  const outShape = [];
  const lenA = shapeA.length;
  const lenB = shapeB.length;
  const outlen = Math.max(lenA, lenB);
  for (let i = 0; i < outlen; ++i) {
    let a = shapeA[lenA - i - 1];
    if (a === undefined) {
      a = 1;
    }
    let b = shapeB[lenB - i - 1];
    if (b === undefined) {
      b = 1;
    }
    if (a === 1) {
      outShape.unshift(b);
    } else if (b === 1) {
      outShape.unshift(a);
    } else if (a !== b) {
      throw new Error(`Shapes [${shapeA}] and [${shapeB}] are incompatible.`);
    } else {
      outShape.unshift(a);
    }
  }
  return outShape;
}
