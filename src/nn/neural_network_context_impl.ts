import {Constant} from './constant';
import {Input} from './input';
import {Model} from './model_impl';
import {NamedOperand} from './named_operand';
import {NeuralNetworkContext as NeuralNetworkContextInterface} from './neural_network_context';
import {Operand} from './operand';
import {OperandDescriptor} from './operand_descriptor';
import {OperandLayout} from './operand_layout';
import {OperandType} from './operand_type';
import {Add} from './ops/add';
import {AveragePool2d} from './ops/average_pool2d';
import {Conv2d} from './ops/conv2d';
import {MatMul} from './ops/matmul';
import {MaxPool2d} from './ops/max_pool2d';
import {Mul} from './ops/mul';
import {Relu} from './ops/relu';
import {Reshape} from './ops/reshape';
import {Softmax} from './ops/softmax';
import {Transpose} from './ops/transpose';
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
