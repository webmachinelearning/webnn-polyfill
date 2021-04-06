import * as tf from '@tensorflow/tfjs-core';

import {Operand, OperandDescriptor, OperandType} from './operand';
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

export function isIntegerArray(array: number[]): boolean {
  return array instanceof Array && array.every(v => isInteger(v));
}

export function isTypedArray(array: TypedArray): boolean {
  return array instanceof Float32Array || array instanceof Int32Array ||
      array instanceof Uint32Array || array instanceof Int16Array ||
      array instanceof Uint16Array || array instanceof Int8Array ||
      array instanceof Uint8Array;
}

export function getTypedArray(type: OperandType): Float32ArrayConstructor|
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

export function getDataType(type: OperandType): tf.DataType {
  if (type === 'float32') {
    return 'float32';
  } else if (type === 'int32') {
    return 'int32';
  } else {
    throw new Error('The operand type is not supported by TF.js.');
  }
}

export function createOperandDescriptorFromTensor(tensor: tf.Tensor):
    OperandDescriptor {
  let type: OperandType;
  if (tensor.dtype === 'float32') {
    type = OperandType.float32;
  } else if (tensor.dtype === 'int32') {
    type = OperandType.int32;
  }
  return {type, dimensions: tensor.shape} as OperandDescriptor;
}

export function validateOperandDescriptor(desc: OperandDescriptor): void {
  assert(desc.type in OperandType, 'The operand type is invalid.');
  if (desc.dimensions) {
    assert(isIntegerArray(desc.dimensions), 'The dimensions is invalid.');
  }
}

export function isDyanmicShape(dimensions: number[]): boolean {
  return !dimensions.every(x => x > 0);
}

export function validateTypedArray(
    value: TypedArray, type: OperandType, dimensions: number[]): void {
  assert(isTypedArray(value), 'The value is not a typed array.');
  assert(value instanceof getTypedArray(type), 'The type of value is invalid.');
  assert(
      value.length === sizeFromDimensions(dimensions),
      `the value length ${value.length} is invalid, size of ` +
          `[${dimensions}] ${sizeFromDimensions(dimensions)} ` +
          'is expected.');
}

export function validateValueType(value: number, type: OperandType): void {
  if (type === OperandType.int32) {
    assert(Number.isInteger(value), 'the value is not an int32.');
  } else if (type === OperandType.uint32) {
    assert(
        Number.isInteger(value) && value >= 0, 'the value is not an uint32.');
  } else if (type === OperandType.int8) {
    assert(
        Number.isInteger(value) && value >= -128 && value <= 127,
        'the value is not an int8.');
  } else if (type === OperandType.uint8) {
    assert(
        Number.isInteger(value) && value >= 0 && value <= 255,
        'the value is not an uint8.');
  }
}

export function createTensor(
    desc: OperandDescriptor, value: TypedArray|number): tf.Tensor {
  const dtype: tf.DataType = getDataType(desc.type);
  if (desc.dimensions !== undefined) {
    validateTypedArray(value as TypedArray, desc.type, desc.dimensions);
    return tf.tensor(value as TypedArray, desc.dimensions, dtype);
  } else {
    if (typeof value === 'number') {
      validateValueType(value, desc.type);
      return tf.scalar(value, dtype);
    } else {
      validateTypedArray(value, desc.type, desc.dimensions);
      return tf.scalar(value[0], dtype);
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

export function validateOperand(input: Operand, name = ''): void {
  assert(input instanceof Operand, `The parameter ${name} is not an operand.`);
}

export function validateOptionalOperand(input: Operand, name = ''): void {
  assert(
      input === undefined || input instanceof Operand,
      `The parameter ${name} is not an optional operand.`);
}

export function validateAxes(axes: number[], rank: number): boolean {
  if (typeof axes !== 'undefined' && axes.length > 0) {
    for (let i = 0; i < axes.length; ++i) {
      if (axes[i] >= rank || axes[i] < -rank) {
        return false;
      }
    }
  }
  return true;
}
