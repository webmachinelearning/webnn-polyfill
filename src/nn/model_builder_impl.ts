import {ConstantOperand} from './constant_operand';
import {Conv2dOptions} from './conv2d_options';
import {GruCellOptions, GruOptions} from './gru_options';
import {InputOperand} from './input_operand';
import {ModelBuilder as ModelBuilderInterface} from './model_builder';
import {Model} from './model_impl';
import {NamedOperands} from './named_operands';
import {OperandDescriptor} from './operand_descriptor';
import {Operand} from './operand_impl';
import {OperandType} from './operand_type';
import {Add, Div, Max, Min, Mul, Sub} from './ops/binary';
import {Concat} from './ops/concat';
import {Conv2d} from './ops/conv2d';
import {Gru, GruCell} from './ops/gru';
import {MatMul} from './ops/matmul';
import {AveragePool2d, MaxPool2d} from './ops/pool2d';
import {Reshape} from './ops/reshape';
import {Slice} from './ops/slice';
import {Softmax} from './ops/softmax';
import {Squeeze} from './ops/squeeze';
import {Transpose} from './ops/transpose';
import {Exp, Relu, Sigmoid, Sqrt, Tanh} from './ops/unary';
import {Pooling2dOptions} from './pooling2d_options';
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
  constant(value: number, type?: OperandType): ConstantOperand;
  constant(
      descOrValue: OperandDescriptor|number,
      valueOrType: TypedArray|OperandType): ConstantOperand {
    if (typeof descOrValue === 'number') {
      if (valueOrType === undefined) {
        valueOrType = OperandType.float32;
      }
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

  // element-wise binary operations
  add(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new Add(a, b)).output;
  }

  sub(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new Sub(a, b)).output;
  }

  mul(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new Mul(a, b)).output;
  }

  div(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new Div(a, b)).output;
  }

  max(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new Max(a, b)).output;
  }

  min(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new Min(a, b)).output;
  }

  // element-wise unary operations
  exp(x: Operand): Operand {
    this.validateOperandBuilder([x]);
    return (new Exp(x)).output;
  }

  sigmoid(x: Operand): Operand {
    this.validateOperandBuilder([x]);
    return (new Sigmoid(x)).output;
  }

  sqrt(x: Operand): Operand {
    this.validateOperandBuilder([x]);
    return (new Sqrt(x)).output;
  }

  tanh(x: Operand): Operand {
    this.validateOperandBuilder([x]);
    return (new Tanh(x)).output;
  }

  concat(inputs: Operand[], axis: number): Operand {
    this.validateOperandBuilder(inputs);
    return (new Concat(inputs, axis)).output;
  }

  conv2d(input: Operand, filter: Operand, options: Conv2dOptions = {}):
      Operand {
    this.validateOperandBuilder([input, filter]);
    return (new Conv2d(
                input, filter, options.padding, options.strides,
                options.dilations, options.groups, options.layout))
        .output;
  }

  gru(input: Operand, weight: Operand, recurrentWeight: Operand, steps: number,
      hiddenSize: number, options: GruOptions = {}): Operand[] {
    this.validateOperandBuilder([
      input, weight, recurrentWeight, options.bias, options.recurrentBias,
      options.initialHiddenState
    ]);
    return Gru.build(
        this, input, weight, recurrentWeight, steps, hiddenSize, options.bias,
        options.recurrentBias, options.initialHiddenState, options.resetAfter,
        options.returnSequence, options.direction, options.layout,
        options.activations);
  }

  gruCell(
      input: Operand, weight: Operand, recurrentWeight: Operand,
      hiddenState: Operand, hiddenSize: number,
      options: GruCellOptions = {}): Operand {
    this.validateOperandBuilder([
      input, weight, recurrentWeight, hiddenState, options.bias,
      options.recurrentBias
    ]);
    return GruCell.build(
        this, input, weight, recurrentWeight, hiddenState, hiddenSize,
        options.bias, options.recurrentBias, options.resetAfter, options.layout,
        options.activations);
  }

  matmul(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new MatMul(a, b)).output;
  }

  // pooling operations
  averagePool2d(input: Operand, options: Pooling2dOptions = {}): Operand {
    this.validateOperandBuilder([input]);
    return (new AveragePool2d(
                input, options.windowDimensions, options.padding,
                options.strides, options.dilations, options.layout))
        .output;
  }

  maxPool2d(input: Operand, options: Pooling2dOptions = {}): Operand {
    this.validateOperandBuilder([input]);
    return (new MaxPool2d(
                input, options.windowDimensions, options.padding,
                options.strides, options.dilations, options.layout))
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

  slice(input: Operand, starts: number[], sizes: number[], axes?: number[]):
      Operand {
    this.validateOperandBuilder([input]);
    return (new Slice(input, starts, sizes, axes)).output;
  }

  softmax(x: Operand): Operand {
    this.validateOperandBuilder([x]);
    return (new Softmax(x)).output;
  }

  squeeze(input: Operand, axes?: number[]): Operand {
    this.validateOperandBuilder([input]);
    return (new Squeeze(input, axes)).output;
  }

  transpose(input: Operand, permutation?: number[]): Operand {
    this.validateOperandBuilder([input]);
    return (new Transpose(input, permutation)).output;
  }
}
