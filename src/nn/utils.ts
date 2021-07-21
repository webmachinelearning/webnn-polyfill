import * as tf from '@tensorflow/tfjs-core';
import {ExplicitPadding} from '@tensorflow/tfjs-core/src/ops/conv_util';

import {MLAutoPad, MLBufferView} from './graph_builder';
import {MLOperand, MLOperandDescriptor, MLOperandType} from './operand';
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

export function isTypedArray(array: MLBufferView|WebGLTexture): boolean {
  return array instanceof Float32Array || array instanceof Int32Array ||
      array instanceof Uint32Array || array instanceof Int16Array ||
      array instanceof Uint16Array || array instanceof Int8Array ||
      array instanceof Uint8Array;
}

export function getTypedArray(type: MLOperandType): Float32ArrayConstructor|
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

export function getDataType(type: MLOperandType): tf.DataType {
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
  let type: MLOperandType;
  if (tensor.dtype === 'float32') {
    type = MLOperandType.float32;
  } else if (tensor.dtype === 'int32') {
    type = MLOperandType.int32;
  }
  return {type, dimensions: tensor.shape} as MLOperandDescriptor;
}

export function validateOperandDescriptor(desc: MLOperandDescriptor): void {
  assert(desc.type in MLOperandType, 'The operand type is invalid.');
  if (desc.dimensions) {
    assert(isIntegerArray(desc.dimensions), 'The dimensions is invalid.');
  }
}

export function isDyanmicShape(dimensions: number[]): boolean {
  return !dimensions.every(x => x > 0);
}

export function validateTypedArray(
    value: TypedArray, type: MLOperandType, dimensions: number[]): void {
  assert(isTypedArray(value), 'The value is not a typed array.');
  assert(value instanceof getTypedArray(type), 'The type of value is invalid.');
  assert(
      value.length === sizeFromDimensions(dimensions),
      `the value length ${value.length} is invalid, size of ` +
          `[${dimensions}] ${sizeFromDimensions(dimensions)} ` +
          'is expected.');
}

export function validateValueType(value: number, type: MLOperandType): void {
  if (type === MLOperandType.int32) {
    assert(Number.isInteger(value), 'the value is not an int32.');
  } else if (type === MLOperandType.uint32) {
    assert(
        Number.isInteger(value) && value >= 0, 'the value is not an uint32.');
  } else if (type === MLOperandType.int8) {
    assert(
        Number.isInteger(value) && value >= -128 && value <= 127,
        'the value is not an int8.');
  } else if (type === MLOperandType.uint8) {
    assert(
        Number.isInteger(value) && value >= 0 && value <= 255,
        'the value is not an uint8.');
  }
}

export function createTensor(
    desc: MLOperandDescriptor,
    value: MLBufferView|WebGLTexture|number): tf.Tensor {
  const dtype: tf.DataType = getDataType(desc.type);
  if (desc.dimensions !== undefined) {
    assert(
        isTypedArray(value as MLBufferView | WebGLTexture),
        'Only ArrayBufferView value is supported.');
    const array = value as TypedArray;
    validateTypedArray(array, desc.type, desc.dimensions);
    const clonedArray = cloneTypedArray(array);
    return tf.tensor(clonedArray, desc.dimensions, dtype);
  } else {
    if (typeof value === 'number') {
      validateValueType(value, desc.type);
      return tf.scalar(value, dtype);
    } else {
      validateTypedArray(value as TypedArray, desc.type, desc.dimensions);
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
      if (axes[i] >= rank || axes[i] < -rank) {
        return false;
      }
    }
  }
  return true;
}

export function getPaddings(
    input: tf.Tensor4D, filter: tf.Tensor4D,
    padding: [number, number, number, number], strides: [number, number],
    outputPadding: [number, number], dilations: [number, number],
    autoPad: MLAutoPad): 'valid'|'same'|ExplicitPadding {
    // WebNN padding:
    //   [beginning_height, ending_height, beginning_width, ending_width]
    // tf.conv2d NHWC should be in the following form:
    //   [[0, 0], [pad_top,pad_bottom], [pad_left, pad_right], [0, 0]]      
    let resultPadding: 'valid'|'same'|ExplicitPadding;
    if (autoPad === MLAutoPad.explicit) {
      if (padding.every(v => v === 0)) {
        resultPadding = 'valid';
      } else {
        resultPadding = [
          [0, 0], [padding[0], padding[1]],
          [padding[2], padding[3]], [0, 0]
        ] as ExplicitPadding;
      }
    } else {
      if (autoPad === MLAutoPad['same-upper']) {
        resultPadding = 'same';
      } else {
        // Calculate the explicit paddings for 'same-lower'
        resultPadding = [[0, 0], [0, 0], [0, 0], [0, 0]];
        const outputSizes = [0, 0];
        for (let i = 0; i < 2; ++i) {
          outputSizes[i] = Math.ceil(input.shape[1 + i] / strides[i]);
        }
        const totalPadding: [number, number] = [0, 0];
        for (let i = 0; i < 2; ++i) {
          totalPadding[i] = strides[i] * (outputSizes[i] - 1) +
              outputPadding[i] +
              ((filter.shape[i] - 1) * dilations[i] + 1) -
              input.shape[1 + i];
        }
        for (let i = 0; i < 2; ++i) {
          resultPadding[i + 1][0] =
              totalPadding[i] - Math.floor(totalPadding[i] / 2);
              resultPadding[i + 1][1] = Math.floor(totalPadding[i] / 2);
        }
      }
    }
    return resultPadding;
}