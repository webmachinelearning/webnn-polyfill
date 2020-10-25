import {ConstantOperand} from './constant_operand';
import {InputOperand} from './input_operand';
import {ModelBuilder as ModelBuilderInterface} from './model_builder';
import {Model} from './model_impl';
import {NamedOperands} from './named_operands';
import {OperandDescriptor} from './operand_descriptor';
import {Operand} from './operand_impl';
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
import * as utils from './utils';

export class ModelBuilder implements ModelBuilderInterface {
  createModel(outputs: NamedOperands): Model {
    return new Model(outputs);
  }

  input(name: string, desc: OperandDescriptor): InputOperand {
    return new InputOperand(name, desc, this);
  }

  constant(desc: OperandDescriptor, value: TypedArray): ConstantOperand;
  constant(value: number, type: OperandType): ConstantOperand;
  constant(
      descOrValue: OperandDescriptor|number,
      valueOrType: TypedArray|OperandType): ConstantOperand {
    if (typeof descOrValue === 'number') {
      return ConstantOperand.createScalar(
          descOrValue, valueOrType as OperandType, this);
    } else {
      return ConstantOperand.createTensor(
          descOrValue, valueOrType as TypedArray, this);
    }
  }

  private validateOperandBuilder(operands: Operand[]) {
    utils.assert(
        operands.every(operand => operand ? operand.builder === this : true),
        'The operand is not built by this builder.');
  }

  add(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new Add(a, b)).output;
  }

  averagePool2d(
      input: Operand, windowDimensions: [number, number] = [-1, -1],
      padding: [number, number, number, number] = [0, 0, 0, 0],
      strides: [number, number] = [1, 1], dilations: [number, number] = [1, 1],
      layout: OperandLayout = OperandLayout.nchw): Operand {
    this.validateOperandBuilder([input]);
    return (new AveragePool2d(
                input, windowDimensions, padding, strides, dilations, layout))
        .output;
  }

  conv2d(
      input: Operand, filter: Operand,
      padding: [number, number, number, number] = [0, 0, 0, 0],
      strides: [number, number] = [1, 1], dilations: [number, number] = [1, 1],
      groups = 1, layout: OperandLayout = OperandLayout.nchw): Operand {
    this.validateOperandBuilder([input, filter]);
    return (new Conv2d(
                input, filter, padding, strides, dilations, groups, layout))
        .output;
  }

  matmul(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new MatMul(a, b)).output;
  }

  mul(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new Mul(a, b)).output;
  }

  maxPool2d(
      input: Operand, windowDimensions: [number, number] = [-1, -1],
      padding: [number, number, number, number] = [0, 0, 0, 0],
      strides: [number, number] = [1, 1], dilations: [number, number] = [1, 1],
      layout: OperandLayout = OperandLayout.nchw): Operand {
    this.validateOperandBuilder([input]);
    return (new MaxPool2d(
                input, windowDimensions, padding, strides, dilations, layout))
        .output;
  }

  relu(input: Operand): Operand {
    this.validateOperandBuilder([input]);
    return (new Relu(input)).output;
  }

  reshape(input: Operand, newShape: number[]): Operand {
    this.validateOperandBuilder([input]);
    return (new Reshape(input, newShape)).output;
  }

  softmax(x: Operand): Operand {
    this.validateOperandBuilder([x]);
    return (new Softmax(x)).output;
  }

  transpose(input: Operand, permutation?: number[]): Operand {
    this.validateOperandBuilder([input]);
    return (new Transpose(input, permutation)).output;
  }
}
