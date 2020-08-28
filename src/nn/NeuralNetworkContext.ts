import { OperandDescriptor } from './OperandDescriptor';
import { Input } from './Input';
import { Model } from './Model';
import { Constant } from './Constant';
import { Operand } from './Operand';
import { Add } from './ops/Add';
import { Mul } from './ops/Mul';
import { OperandType } from './OperandType';
import { NamedOperand } from './NamedOperand';
import { OperandLayout } from './OperandLayout';
import { Conv2d } from './ops/Conv2d';
import { AveragePool2d } from './ops/AveragePool2d';
import { MaxPool2d } from './ops/MaxPool2d';
import { Reshape } from './ops/Reshape';
import { Relu } from './ops/Relu';
import { MatMul } from './ops/MatMul';
import { Softmax } from './ops/Softmax';
import { Transpose } from './ops/Transpose';
import { TypedArray } from './utils';

/**
 * Implements the [NeuralNetworkContext](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext) interface.
 */
export class NeuralNetworkContext {
  /** */
  async createModel(outputs: NamedOperand[]): Promise<Model> {
    return new Model(outputs);
  }

  /** */
  input(name: string, desc: OperandDescriptor): Input {
    return new Input(name, desc);
  }

  /** */
  constant(desc: OperandDescriptor, value: TypedArray): Constant;
  /** */
  constant(value: number, type: OperandType): Constant;
  constant(descOrValue: OperandDescriptor|number,
           valueOrType: TypedArray|OperandType): Constant {
    if (typeof descOrValue === 'number') {
      return Constant.createScalar(descOrValue, valueOrType as OperandType);
    } else {
      return Constant.createTensor(descOrValue, valueOrType as TypedArray);
    }
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-binary)
   */
  add(a: Operand, b: Operand): Operand {
    return (new Add(a, b)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-pool2d)
   */
  averagePool2d(input: Operand,
                windowDimensions: [number, number] = [-1, -1],
                padding: [number, number, number, number] = [0, 0, 0, 0],
                strides: [number, number] = [1, 1],
                dilations: [number, number] = [1, 1],
                layout: OperandLayout = OperandLayout.nchw): Operand {
    return (new AveragePool2d(input, windowDimensions, padding, strides,
                              dilations, layout)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-conv2d)
   */
  conv2d(input: Operand, filter: Operand,
         padding: [number, number, number, number] = [0, 0, 0, 0],
         strides: [number, number] = [1, 1],
         dilations: [number, number] = [1, 1],
         groups = 1,
         layout: OperandLayout = OperandLayout.nchw): Operand {
    return (new Conv2d(input, filter, padding, strides, dilations, groups,
                       layout)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-matmul)
   */
  matmul(a: Operand, b: Operand): Operand {
    return (new MatMul(a, b)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-binary)
   */
  mul(a: Operand, b: Operand): Operand {
    return (new Mul(a, b)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-pool2d)
   */
  maxPool2d(input: Operand,
            windowDimensions: [number, number] = [-1, -1],
            padding: [number, number, number, number] = [0, 0, 0, 0],
            strides: [number, number] = [1, 1],
            dilations: [number, number] = [1, 1],
            layout: OperandLayout = OperandLayout.nchw): Operand {
    return (new MaxPool2d(input, windowDimensions, padding, strides, dilations,
                          layout)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-relu)
   */
  relu(input: Operand): Operand {
    return (new Relu(input)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-reshape)
   */
  reshape(input: Operand, newShape: number[]): Operand {
    return (new Reshape(input, newShape)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-softmax)
   */
  softmax(x: Operand): Operand {
    return (new Softmax(x)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-neuralnetworkcontext-transpose)
   */
  transpose(input: Operand, permutation?: number[]): Operand {
    return (new Transpose(input, permutation)).output;
  }
}
