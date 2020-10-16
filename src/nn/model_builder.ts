import {ModelBuilder as ModelBuilderImpl} from './model_builder_impl';
import {Model} from './model';
import {NamedOperands} from './named_operands';
import {Operand} from './operand';
import {OperandDescriptor} from './operand_descriptor';
import {OperandLayout} from './operand_layout';
import {OperandType} from './operand_type';
import {ArrayBufferView} from './types';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder)
 */
export interface ModelBuilder {
  /** */
  createModel(outputs: NamedOperands): Model;

  /** */
  input(name: string, desc: OperandDescriptor): Operand;

  /** */
  constant(desc: OperandDescriptor, value: ArrayBufferView): Operand;
  /** */
  constant(value: number, type: OperandType): Operand;

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-binary)
   */
  add(a: Operand, b: Operand): Operand;

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-pool2d)
   */
  averagePool2d(
      input: Operand, windowDimensions?: [number, number],
      padding?: [number, number, number, number], strides?: [number, number],
      dilations?: [number, number], layout?: OperandLayout): Operand;

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-conv2d)
   */
  conv2d(
      input: Operand, filter: Operand,
      padding?: [number, number, number, number], strides?: [number, number],
      dilations?: [number, number], groups?: number,
      layout?: OperandLayout): Operand;

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-matmul)
   */
  matmul(a: Operand, b: Operand): Operand;

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-binary)
   */
  mul(a: Operand, b: Operand): Operand;

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-pool2d)
   */
  maxPool2d(
      input: Operand, windowDimensions?: [number, number],
      padding?: [number, number, number, number], strides?: [number, number],
      dilations?: [number, number], layout?: OperandLayout): Operand;

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-relu)
   */
  relu(input: Operand): Operand;

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-reshape)
   */
  reshape(input: Operand, newShape: number[]): Operand;

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-softmax)
   */
  softmax(x: Operand): Operand;

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-transpose)
   */
  transpose(input: Operand, permutation?: number[]): Operand;
}

interface ModelBuilderConstructor {
  new(): ModelBuilder;
}
// eslint-disable-next-line no-redeclare
export const ModelBuilder: ModelBuilderConstructor = ModelBuilderImpl;
