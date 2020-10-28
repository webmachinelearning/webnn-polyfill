import {Model} from './model';
import {ConstantOperand, InputOperand, Operand} from './operand';
import {Add, Div, Max, Min, Mul, Sub} from './ops/binary';
import {Concat} from './ops/concat';
import {Conv2d} from './ops/conv2d';
import {Exp, Relu, Sigmoid, Sqrt, Tanh} from './ops/unary';
import {ArrayBufferView as TypedArray} from './types';
import {OperandDescriptor, OperandType} from './operand';
import {Gru, GruCell} from './ops/gru';
import {MatMul} from './ops/matmul';
import {AveragePool2d, MaxPool2d} from './ops/pool2d';
import {Reshape} from './ops/reshape';
import {Slice} from './ops/slice';
import {Softmax} from './ops/softmax';
import {Squeeze} from './ops/squeeze';
import {Transpose} from './ops/transpose';
import * as utils from './utils';

/**
 * [OperandLayout](https://webmachinelearning.github.io/webnn/#enumdef-operandlayout)
 */
export enum OperandLayout {
  'nchw' = 'nchw',
  'nhwc' = 'nhwc'
}

/**
 * [Conv2dOptions](https://webmachinelearning.github.io/webnn/#dictdef-conv2doptions)
 */
export interface Conv2dOptions {
  /** */
  padding?: [number, number, number, number];
  /** */
  strides?: [number, number];
  /** */
  dilations?: [number, number];
  /** */
  groups?: number;
  /** */
  layout?: OperandLayout;
}

/**
 * [RecurrentNetworkWeightLayout](https://webmachinelearning.github.io/webnn/#enumdef-recurrentnetworkweightlayout)
 */
export enum RecurrentNetworkWeightLayout {
  'zrn' = 'zrn',
  'rzn' = 'rzn',
}

/**
 * [RecurrentNetworkActivation](https://webmachinelearning.github.io/webnn/#enumdef-recurrentnetworkactivation)
 */
export enum RecurrentNetworkActivation {
  'relu' = 'relu',
  'sigmoid' = 'sigmoid',
  'tanh' = 'tanh',
}

/**
 * [RecurrentNetworkDirection](https://webmachinelearning.github.io/webnn/#enumdef-recurrentnetworkdirection)
 */
export enum RecurrentNetworkDirection {
  'forward' = 'forward',
  'backward' = 'backward',
  'both' = 'both',
}

/**
 * [GruOptions](https://webmachinelearning.github.io/webnn/#dictdef-gruoptions)
 */
export interface GruOptions {
  /** */
  bias?: Operand;
  /** */
  recurrentBias?: Operand;
  /** */
  initialHiddenState?: Operand;
  /** */
  resetAfter?: boolean;
  /** */
  returnSequence?: boolean;
  /** */
  direction?: RecurrentNetworkDirection;
  /** */
  layout?: RecurrentNetworkWeightLayout;
  /** */
  activations?: RecurrentNetworkActivation[];
}

/**
 * [GruCellOptions](https://webmachinelearning.github.io/webnn/#dictdef-grucelloptions)
 */
export interface GruCellOptions {
  /** */
  bias?: Operand;
  /** */
  recurrentBias?: Operand;
  /** */
  resetAfter?: boolean;
  /** */
  layout?: RecurrentNetworkWeightLayout;
  /** */
  activations?: RecurrentNetworkActivation[];
}

/**
 * [Pooling2dOptions](https://webmachinelearning.github.io/webnn/#dictdef-pool2doptions)
 */
export interface Pooling2dOptions {
  /** */
  windowDimensions?: [number, number];
  /** */
  padding?: [number, number, number, number];
  /** */
  strides?: [number, number];
  /** */
  dilations?: [number, number];
  /** */
  layout?: OperandLayout;
}

/**
 * [SliceOptions](https://webmachinelearning.github.io/webnn/#dictdef-sliceoptions)
 */
export interface SliceOptions {
  /** */
  axes?: number[];
}

/**
 * [NamedOperands](https://webmachinelearning.github.io/webnn/#typedefdef-namedoperands)
 */
export type NamedOperands = Record<string, Operand>;

/**
 * [ModelBuilder](https://webmachinelearning.github.io/webnn/#api-modelbuilder)
 */
export class ModelBuilder {
  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder)
  */
  createModel(outputs: NamedOperands): Model {
    return new Model(outputs);
  }

  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder)
  */
  input(name: string, desc: OperandDescriptor): InputOperand {
    return new InputOperand(name, desc, this);
  }

  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder)
  */
  constant(desc: OperandDescriptor, value: TypedArray): ConstantOperand;
  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder)
  */
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
  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-binary)
  */
  add(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new Add(a, b)).output;
  }

  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-binary)
  */
  sub(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new Sub(a, b)).output;
  }

  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-binary)
  */
  mul(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new Mul(a, b)).output;
  }

  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-binary)
  */
  div(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new Div(a, b)).output;
  }

  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-binary)
  */
  max(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new Max(a, b)).output;
  }

  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-binary)
  */
  min(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new Min(a, b)).output;
  }

  // element-wise unary operations
  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-unary)
  */
  exp(x: Operand): Operand {
    this.validateOperandBuilder([x]);
    return (new Exp(x)).output;
  }

  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-unary)
  */
  sigmoid(x: Operand): Operand {
    this.validateOperandBuilder([x]);
    return (new Sigmoid(x)).output;
  }

  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-unary)
  */
  sqrt(x: Operand): Operand {
    this.validateOperandBuilder([x]);
    return (new Sqrt(x)).output;
  }

  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-unary)
  */
  tanh(x: Operand): Operand {
    this.validateOperandBuilder([x]);
    return (new Tanh(x)).output;
  }

  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-concat)
  */
  concat(inputs: Operand[], axis: number): Operand {
    this.validateOperandBuilder(inputs);
    return (new Concat(inputs, axis)).output;
  }

  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-conv2d)
  */
  conv2d(input: Operand, filter: Operand, options: Conv2dOptions = {}):
      Operand {
    this.validateOperandBuilder([input, filter]);
    return (new Conv2d(
                input, filter, options.padding, options.strides,
                options.dilations, options.groups, options.layout))
        .output;
  }

  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-gru)
  */
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

  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-grucell)
  */
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

  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-matmul)
  */
  matmul(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new MatMul(a, b)).output;
  }

  // pooling operations
  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-pool2d)
  */
  averagePool2d(input: Operand, options: Pooling2dOptions = {}): Operand {
    this.validateOperandBuilder([input]);
    return (new AveragePool2d(
                input, options.windowDimensions, options.padding,
                options.strides, options.dilations, options.layout))
        .output;
  }

  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-pool2d)
  */
  maxPool2d(input: Operand, options: Pooling2dOptions = {}): Operand {
    this.validateOperandBuilder([input]);
    return (new MaxPool2d(
                input, options.windowDimensions, options.padding,
                options.strides, options.dilations, options.layout))
        .output;
  }

  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-relu)
  */
  relu(input: Operand): Operand {
    this.validateOperandBuilder([input]);
    return (new Relu(input)).output;
  }

  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-reshape)
  */
  reshape(input: Operand, newShape: number[]): Operand {
    this.validateOperandBuilder([input]);
    return (new Reshape(input, newShape)).output;
  }

  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-slice)
  */
  slice(
      input: Operand, starts: number[], sizes: number[],
      options: SliceOptions = {}): Operand {
    this.validateOperandBuilder([input]);
    return (new Slice(input, starts, sizes, options.axes)).output;
  }

  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-softmax)
  */
  softmax(x: Operand): Operand {
    this.validateOperandBuilder([x]);
    return (new Softmax(x)).output;
  }

  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-squeeze)
  */
  squeeze(input: Operand, axes?: number[]): Operand {
    this.validateOperandBuilder([input]);
    return (new Squeeze(input, axes)).output;
  }

  /** 
   * [spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-transpose)
  */
  transpose(input: Operand, permutation?: number[]): Operand {
    this.validateOperandBuilder([input]);
    return (new Transpose(input, permutation)).output;
  }
}
