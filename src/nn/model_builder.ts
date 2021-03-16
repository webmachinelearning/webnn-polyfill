import {Model} from './model';
import {ConstantOperand, InputOperand, Operand, OperandDescriptor, OperandType} from './operand';
import {BatchNormalization} from './ops/batch_norm';
import {Add, Div, MatMul, Max, Min, Mul, Pow, Sub} from './ops/binary';
import {Clamp} from './ops/clamp';
import {Concat} from './ops/concat';
import {Conv2d} from './ops/conv2d';
import {Gemm} from './ops/gemm';
import {Gru, GruCell} from './ops/gru';
import {InstanceNormalization} from './ops/instance_norm';
import {LeakyRelu} from './ops/leaky_relu';
import {Pad} from './ops/pad';
import {AveragePool2d, MaxPool2d} from './ops/pool2d';
import {ReduceLogSumExp, ReduceMax, ReduceMean, ReduceMin, ReduceProduct, ReduceSum} from './ops/reduce';
import {Resample} from './ops/resample';
import {Reshape} from './ops/reshape';
import {Slice} from './ops/slice';
import {Softmax} from './ops/softmax';
import {Split} from './ops/split';
import {Squeeze} from './ops/squeeze';
import {Transpose} from './ops/transpose';
import {Exp, Relu, Sigmoid, Tanh} from './ops/unary';
import {ArrayBufferView as TypedArray} from './types';
import * as utils from './utils';

/**
 * [API spec](https://webmachinelearning.github.io/webnn/#enumdef-operandlayout)
 */
export enum InputOperandLayout {
  'nchw' = 'nchw',
  'nhwc' = 'nhwc'
}

/**
 * [API
 * spec](https://webmachinelearning.github.io/webnn/#enumdef-filteroperandlayout)
 */
export enum FilterOperandLayout {
  'oihw' = 'oihw',
  'hwio' = 'hwio',
  'ohwi' = 'ohwi',
}

/**
 * [API
 * spec](https://webmachinelearning.github.io/webnn/#enumdef-autopad)
 */
export enum AutoPad {
  'explicit' = 'explicit',
  'same-upper' = 'same-upper',
  'same-lower' = 'same-lower',
}

/**
 * [API
 * spec](https://webmachinelearning.github.io/webnn/#dictdef-batchnormalizationoptions)
 */
export interface BatchNormalizationOptions {
  scale?: Operand;
  bias?: Operand;
  axis?: number;
  epsilon?: number;
}

/**
 * [API spec](https://webmachinelearning.github.io/webnn/#dictdef-conv2doptions)
 */
export interface Conv2dOptions {
  padding?: [number, number, number, number];
  strides?: [number, number];
  dilations?: [number, number];
  outputPadding?: [number, number];
  outputSizes?: [number, number];
  autoPad?: AutoPad;
  transpose?: boolean;
  groups?: number;
  inputLayout?: InputOperandLayout;
  filterLayout?: FilterOperandLayout;
}

/**
 * [API spec](https://webmachinelearning.github.io/webnn/#dictdef-gemmoptions)
 */
export interface GemmOptions {
  c?: Operand;
  alpha?: number;
  beta?: number;
  aTranspose?: boolean;
  bTranspose?: boolean;
}

/**
 * [API
 * spec](https://webmachinelearning.github.io/webnn/#enumdef-recurrentnetworkweightlayout)
 */
export enum RecurrentNetworkWeightLayout {
  'zrn' = 'zrn',
  'rzn' = 'rzn',
}

/**
 * [API
 * spec](https://webmachinelearning.github.io/webnn/#enumdef-recurrentnetworkactivation)
 */
export enum RecurrentNetworkActivation {
  'relu' = 'relu',
  'sigmoid' = 'sigmoid',
  'tanh' = 'tanh',
}

/**
 * [API
 * spec](https://webmachinelearning.github.io/webnn/#enumdef-recurrentnetworkdirection)
 */
export enum RecurrentNetworkDirection {
  'forward' = 'forward',
  'backward' = 'backward',
  'both' = 'both',
}

/**
 * [API spec](https://webmachinelearning.github.io/webnn/#dictdef-gruoptions)
 */
export interface GruOptions {
  bias?: Operand;
  recurrentBias?: Operand;
  initialHiddenState?: Operand;
  resetAfter?: boolean;
  returnSequence?: boolean;
  direction?: RecurrentNetworkDirection;
  layout?: RecurrentNetworkWeightLayout;
  activations?: RecurrentNetworkActivation[];
}

/**
 * [API
 * spec](https://webmachinelearning.github.io/webnn/#dictdef-grucelloptions)
 */
export interface GruCellOptions {
  bias?: Operand;
  recurrentBias?: Operand;
  resetAfter?: boolean;
  layout?: RecurrentNetworkWeightLayout;
  activations?: RecurrentNetworkActivation[];
}

/**
 * [API
 * spec](https://webmachinelearning.github.io/webnn/#dictdef-instancenormalizationoptions)
 */
export interface InstanceNormalizationOptions {
  scale?: Operand;
  bias?: Operand;
  epsilon?: number;
  layout?: InputOperandLayout;
}

/**
 * [API
 * spec](https://webmachinelearning.github.io/webnn/#dictdef-leakyreluoptions)
 */
export interface LeakyReluOptions {
  alpha?: number;
}

/**
 * [API
 * spec](https://webmachinelearning.github.io/webnn/#enumdef-paddingmode)
 */
export enum PaddingMode {
  'constant' = 'constant',
  'edge' = 'edge',
  'reflection' = 'reflection',
  'symmetric' = 'symmetric'
}

/**
 * [API
 * spec](https://webmachinelearning.github.io/webnn/#dictdef-padoptions)
 */
export interface PadOptions {
  mode?: PaddingMode;
  value?: number;
}

/**
 * [API spec](https://webmachinelearning.github.io/webnn/#dictdef-pool2doptions)
 */
export interface Pooling2dOptions {
  windowDimensions?: [number, number];
  padding?: [number, number, number, number];
  strides?: [number, number];
  dilations?: [number, number];
  autoPad?: AutoPad;
  layout?: InputOperandLayout;
}

/**
 * [API
 * spec](https://webmachinelearning.github.io/webnn/#enumdef-interpolationmode)
 */
export enum InterpolationMode {
  'nearest-neighbor' = 'nearest-neighbor',
  'linear' = 'linear'
}

/**
 * [API
 * spec](https://webmachinelearning.github.io/webnn/#dictdef-resampleoptions)
 */
export interface ResampleOptions {
  mode?: InterpolationMode;
  scales?: [number, number, number, number];
  sizes?: [number, number, number, number];
}

/**
 * [API spec](https://webmachinelearning.github.io/webnn/#dictdef-sliceoptions)
 */
export interface SliceOptions {
  axes?: number[];
}

/**
 * [API
 * spec](https://webmachinelearning.github.io/webnn/#dictdef-squeezeoptions)
 */
export interface SqueezeOptions {
  axes?: number[];
}

/**
 * [API
 * spec](https://webmachinelearning.github.io/webnn/#dictdef-transposeoptions)
 */
export interface TransposeOptions {
  permutation?: number[];
}

/**
 * [API spec](https://webmachinelearning.github.io/webnn/#dictdef-clampoptions)
 */
export interface ClampOptions {
  minValue?: Operand;
  maxValue?: Operand;
}

/**
 * [API spec](https://webmachinelearning.github.io/webnn/#dictdef-splitoptions)
 */
export interface SplitOptions {
  axis?: number;
}

/**
 * [API spec](https://webmachinelearning.github.io/webnn/#dictdef-reduceoptions)
 */
export interface ReduceOptions {
  axes?: number[];
  keepDimensions?: boolean;
}

/**
 * [API
 * spec](https://webmachinelearning.github.io/webnn/#typedefdef-namedoperands)
 */
export type NamedOperands = Record<string, Operand>;

/**
 * [API spec](https://webmachinelearning.github.io/webnn/#modelbuilder)
 */
export class ModelBuilder {
  /** @ignore */
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  constructor() {}

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-createmodel)
   */
  createModel(outputs: NamedOperands): Model {
    return new Model(outputs);
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-input)
   */
  input(name: string, desc: OperandDescriptor): Operand {
    return new InputOperand(name, desc, this);
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-constant)
   */
  constant(desc: OperandDescriptor, value: TypedArray): Operand;
  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-constant-value-type)
   */
  constant(value: number, type?: OperandType): Operand;
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

  // element-wise binary operations
  // https://webmachinelearning.github.io/webnn/#dom-modelbuilder-binary
  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-add)
   */
  add(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new Add(a, b)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-sub)
   */
  sub(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new Sub(a, b)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-mul)
   */
  mul(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new Mul(a, b)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-div)
   */
  div(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new Div(a, b)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-max)
   */
  max(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new Max(a, b)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-min)
   */
  min(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new Min(a, b)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-pow)
   *
   */
  pow(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new Pow(a, b)).output;
  }

  // element-wise unary operations
  // https://webmachinelearning.github.io/webnn/#dom-modelbuilder-unary
  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-exp)
   */
  exp(x: Operand): Operand {
    this.validateOperandBuilder([x]);
    return (new Exp(x)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-sigmoid)
   */
  sigmoid(x: Operand): Operand {
    this.validateOperandBuilder([x]);
    return (new Sigmoid(x)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-tanh)
   */
  tanh(x: Operand): Operand {
    this.validateOperandBuilder([x]);
    return (new Tanh(x)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-batchnormalization)
   */
  batchNormalization(
      input: Operand, mean: Operand, variance: Operand,
      options: BatchNormalizationOptions = {}): Operand {
    this.validateOperandBuilder(
        [input, mean, variance, options.scale, options.bias]);
    return (new BatchNormalization(input, mean, variance, options)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-clamp)
   */
  clamp(x: Operand, options: ClampOptions = {}): Operand {
    this.validateOperandBuilder([x, options.minValue, options.maxValue]);
    return (new Clamp(x, options)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-concat)
   */
  concat(inputs: Operand[], axis: number): Operand {
    this.validateOperandBuilder(inputs);
    return (new Concat(inputs, axis)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-conv2d)
   */
  conv2d(input: Operand, filter: Operand, options: Conv2dOptions = {}):
      Operand {
    this.validateOperandBuilder([input, filter]);
    return (new Conv2d(input, filter, options)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-gemm)
   */
  gemm(a: Operand, b: Operand, options: GemmOptions = {}): Operand {
    this.validateOperandBuilder([a, b, options.c]);
    return Gemm.build(this, a, b, options);
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-gru)
   */
  gru(input: Operand, weight: Operand, recurrentWeight: Operand, steps: number,
      hiddenSize: number, options: GruOptions = {}): Operand[] {
    this.validateOperandBuilder([
      input, weight, recurrentWeight, options.bias, options.recurrentBias,
      options.initialHiddenState
    ]);
    return (new Gru(input, weight, recurrentWeight, steps, hiddenSize, options))
        .outputs;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-grucell)
   */
  gruCell(
      input: Operand, weight: Operand, recurrentWeight: Operand,
      hiddenState: Operand, hiddenSize: number,
      options: GruCellOptions = {}): Operand {
    this.validateOperandBuilder([
      input, weight, recurrentWeight, hiddenState, options.bias,
      options.recurrentBias
    ]);
    return (new GruCell(
                input, weight, recurrentWeight, hiddenState, hiddenSize,
                options))
        .output;
  }

  instanceNormalization(
      input: Operand, options: InstanceNormalizationOptions = {}): Operand {
    this.validateOperandBuilder([input, options.bias, options.scale]);
    return (new InstanceNormalization(input, options)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-leakyrelu)
   */
  leakyRelu(x: Operand, options: LeakyReluOptions = {}): Operand {
    this.validateOperandBuilder([x]);
    return (new LeakyRelu(x, options.alpha)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-matmul)
   */
  matmul(a: Operand, b: Operand): Operand {
    this.validateOperandBuilder([a, b]);
    return (new MatMul(a, b)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#api-modelbuilder-pad)
   */
  pad(input: Operand, padding: Operand, options: PadOptions = {}): Operand {
    this.validateOperandBuilder([input, padding]);
    return (new Pad(input, padding, options)).output;
  }

  // pooling operations
  // https://webmachinelearning.github.io/webnn/#dom-modelbuilder-pool2d
  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-averagepool2d)
   */
  averagePool2d(input: Operand, options: Pooling2dOptions = {}): Operand {
    this.validateOperandBuilder([input]);
    return (new AveragePool2d(input, options)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-maxpool2d)
   */
  maxPool2d(input: Operand, options: Pooling2dOptions = {}): Operand {
    this.validateOperandBuilder([input]);
    return (new MaxPool2d(input, options)).output;
  }

  // reduction operations
  // https://webmachinelearning.github.io/webnn/#api-modelbuilder-reduce

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-reducelogsumexp)
   */
  reduceLogSumExp(input: Operand, options: ReduceOptions = {}): Operand {
    this.validateOperandBuilder([input]);
    return (new ReduceLogSumExp(input, options)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-reducemax)
   */
  reduceMax(input: Operand, options: ReduceOptions = {}): Operand {
    this.validateOperandBuilder([input]);
    return (new ReduceMax(input, options)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-reducemean)
   */
  reduceMean(input: Operand, options: ReduceOptions = {}): Operand {
    this.validateOperandBuilder([input]);
    return (new ReduceMean(input, options)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-reducemin)
   */
  reduceMin(input: Operand, options: ReduceOptions = {}): Operand {
    this.validateOperandBuilder([input]);
    return (new ReduceMin(input, options)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-reduceproduct)
   */
  reduceProduct(input: Operand, options: ReduceOptions = {}): Operand {
    this.validateOperandBuilder([input]);
    return (new ReduceProduct(input, options)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-reducesum)
   */
  reduceSum(input: Operand, options: ReduceOptions = {}): Operand {
    this.validateOperandBuilder([input]);
    return (new ReduceSum(input, options)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-relu)
   */
  relu(input: Operand): Operand {
    this.validateOperandBuilder([input]);
    return (new Relu(input)).output;
  }

  resample(input: Operand, options: ResampleOptions = {}): Operand {
    this.validateOperandBuilder([input]);
    return (new Resample(input, options)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-reshape)
   */
  reshape(input: Operand, newShape: number[]): Operand {
    this.validateOperandBuilder([input]);
    return (new Reshape(input, newShape)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-slice)
   */
  slice(
      input: Operand, starts: number[], sizes: number[],
      options: SliceOptions = {}): Operand {
    this.validateOperandBuilder([input]);
    return (new Slice(input, starts, sizes, options.axes)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-softmax)
   */
  softmax(x: Operand): Operand {
    this.validateOperandBuilder([x]);
    return (new Softmax(x)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-split)
   */
  split(input: Operand, splits: number|number[], options: SplitOptions = {}):
      Operand[] {
    this.validateOperandBuilder([input]);
    return (new Split(input, splits, options)).outputs;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-squeeze)
   */
  squeeze(input: Operand, options: SqueezeOptions = {}): Operand {
    this.validateOperandBuilder([input]);
    return (new Squeeze(input, options.axes)).output;
  }

  /**
   * [API
   * spec](https://webmachinelearning.github.io/webnn/#dom-modelbuilder-transpose)
   */
  transpose(input: Operand, options: TransposeOptions = {}): Operand {
    this.validateOperandBuilder([input]);
    return (new Transpose(input, options.permutation)).output;
  }

  private validateOperandBuilder(operands: Operand[]) {
    utils.assert(
        operands.every(
            operand => operand ?
                (operand instanceof Operand && operand.builder === this) :
                true),
        'The operand is not built by this builder.');
  }
}
