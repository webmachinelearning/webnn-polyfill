import {Constant} from './Constant';
import {Input} from './Input';
import {Model} from './ModelImpl';
import {NamedOperand} from './NamedOperand';
import {NeuralNetworkContext as NeuralNetworkContextInterface} from './NeuralNetworkContext';
import {Operand} from './Operand';
import {OperandDescriptor} from './OperandDescriptor';
import {OperandLayout} from './OperandLayout';
import {OperandType} from './OperandType';
import {Add} from './ops/Add';
import {AveragePool2d} from './ops/AveragePool2d';
import {Conv2d} from './ops/Conv2d';
import {MatMul} from './ops/MatMul';
import {MaxPool2d} from './ops/MaxPool2d';
import {Mul} from './ops/Mul';
import {Relu} from './ops/Relu';
import {Reshape} from './ops/Reshape';
import {Softmax} from './ops/Softmax';
import {Transpose} from './ops/Transpose';
import {ArrayBufferView as TypedArray} from './types';

export class NeuralNetworkContext implements NeuralNetworkContextInterface {
  async createModel(outputs: NamedOperand[]): Promise<Model> {
    return new Model(outputs);
  }

  input(name: string, desc: OperandDescriptor): Input {
    return new Input(name, desc);
  }

  constant(desc: OperandDescriptor, value: TypedArray): Constant;
  constant(value: number, type: OperandType): Constant;
  constant(
      descOrValue: OperandDescriptor|number,
      valueOrType: TypedArray|OperandType): Constant {
    if (typeof descOrValue === 'number') {
      return Constant.createScalar(descOrValue, valueOrType as OperandType);
    } else {
      return Constant.createTensor(descOrValue, valueOrType as TypedArray);
    }
  }

  add(a: Operand, b: Operand): Operand {
    return (new Add(a, b)).output;
  }

  averagePool2d(
      input: Operand, windowDimensions: [number, number] = [-1, -1],
      padding: [number, number, number, number] = [0, 0, 0, 0],
      strides: [number, number] = [1, 1], dilations: [number, number] = [1, 1],
      layout: OperandLayout = OperandLayout.nchw): Operand {
    return (new AveragePool2d(
                input, windowDimensions, padding, strides, dilations, layout))
        .output;
  }

  conv2d(
      input: Operand, filter: Operand,
      padding: [number, number, number, number] = [0, 0, 0, 0],
      strides: [number, number] = [1, 1], dilations: [number, number] = [1, 1],
      groups = 1, layout: OperandLayout = OperandLayout.nchw): Operand {
    return (new Conv2d(
                input, filter, padding, strides, dilations, groups, layout))
        .output;
  }

  matmul(a: Operand, b: Operand): Operand {
    return (new MatMul(a, b)).output;
  }

  mul(a: Operand, b: Operand): Operand {
    return (new Mul(a, b)).output;
  }

  maxPool2d(
      input: Operand, windowDimensions: [number, number] = [-1, -1],
      padding: [number, number, number, number] = [0, 0, 0, 0],
      strides: [number, number] = [1, 1], dilations: [number, number] = [1, 1],
      layout: OperandLayout = OperandLayout.nchw): Operand {
    return (new MaxPool2d(
                input, windowDimensions, padding, strides, dilations, layout))
        .output;
  }

  relu(input: Operand): Operand {
    return (new Relu(input)).output;
  }

  reshape(input: Operand, newShape: number[]): Operand {
    return (new Reshape(input, newShape)).output;
  }

  softmax(x: Operand): Operand {
    return (new Softmax(x)).output;
  }

  transpose(input: Operand, permutation?: number[]): Operand {
    return (new Transpose(input, permutation)).output;
  }
}
