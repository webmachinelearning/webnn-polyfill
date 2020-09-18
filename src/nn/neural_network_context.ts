import {Model} from './model';
import {NamedOperand} from './named_operand';
import {NeuralNetworkContext as NeuralNetworkContextImpl} from './neural_network_context_impl';
import {Operand} from './operand';
import {OperandDescriptor} from './operand_descriptor';
import {OperandLayout} from './operand_layout';
import {OperandType} from './operand_type';
import {ArrayBufferView} from './types';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext)
 */
export interface NeuralNetworkContext {
  /** */
  createModel(outputs: NamedOperand[]): Promise<Model>;

  /** */
  input(name: string, desc: OperandDescriptor): Operand;

  /** */
  constant(desc: OperandDescriptor, value: ArrayBufferView): Operand;
  /** */
  constant(value: number, type: OperandType): Operand;

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-binary)
   */
  add(a: Operand, b: Operand): Operand;

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-pool2d)
   */
  averagePool2d(
      input: Operand, windowDimensions?: [number, number],
      padding?: [number, number, number, number], strides?: [number, number],
      dilations?: [number, number], layout?: OperandLayout): Operand;

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-conv2d)
   */
  conv2d(
      input: Operand, filter: Operand,
      padding?: [number, number, number, number], strides?: [number, number],
      dilations?: [number, number], groups?: number,
      layout?: OperandLayout): Operand;

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-matmul)
   */
  matmul(a: Operand, b: Operand): Operand;

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-binary)
   */
  mul(a: Operand, b: Operand): Operand;

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-pool2d)
   */
  maxPool2d(
      input: Operand, windowDimensions?: [number, number],
      padding?: [number, number, number, number], strides?: [number, number],
      dilations?: [number, number], layout?: OperandLayout): Operand;

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-relu)
   */
  relu(input: Operand): Operand;

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-reshape)
   */
  reshape(input: Operand, newShape: number[]): Operand;

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-softmax)
   */
  softmax(x: Operand): Operand;

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-transpose)
   */
  transpose(input: Operand, permutation?: number[]): Operand;
}

interface NeuralNetworkContextConstructor {
  new(): NeuralNetworkContext;
}
// eslint-disable-next-line no-redeclare
export const NeuralNetworkContext: NeuralNetworkContextConstructor =
    NeuralNetworkContextImpl;
