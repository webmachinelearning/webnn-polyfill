import {MLContext} from './context';
import {MLGraph} from './graph';
import {ConstantOperand, InputOperand, MLOperand, MLOperandDescriptor, MLOperandType} from './operand';
import {MLOperator} from './operation';
import {BatchNormalization} from './ops/batch_norm';
import {Add, Div, MatMul, Max, Min, Mul, Pow, Sub} from './ops/binary';
import {Clamp} from './ops/clamp';
import {Concat} from './ops/concat';
import {Conv2d} from './ops/conv2d';
import {ConvTranspose2d} from './ops/conv_transpose2d';
import {Gemm} from './ops/gemm';
import {Gru, GruCell} from './ops/gru';
import {InstanceNormalization} from './ops/instance_norm';
import {LeakyRelu} from './ops/leaky_relu';
import {Pad} from './ops/pad';
import {AveragePool2d, L2Pool2d, MaxPool2d} from './ops/pool2d';
import {ReduceL1, ReduceL2, ReduceLogSumExp, ReduceMax, ReduceMean, ReduceMin, ReduceProduct, ReduceSum} from './ops/reduce';
import {Resample2d} from './ops/resample2d';
import {Reshape} from './ops/reshape';
import {Slice} from './ops/slice';
import {Softmax} from './ops/softmax';
import {Split} from './ops/split';
import {Squeeze} from './ops/squeeze';
import {Transpose} from './ops/transpose';
import {Abs, Ceil, Cos, Exp, Floor, HardSwish, Log, Neg, Relu, Sigmoid, Sin, Tan, Tanh} from './ops/unary';
import {ArrayBufferView} from './types';
import * as utils from './utils';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#enumdef-mlinputoperandlayout)
 */
export enum MLInputOperandLayout {
  'nchw' = 'nchw',
  'nhwc' = 'nhwc'
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-mlbatchnormalizationoptions)
 */
export interface MLBatchNormalizationOptions {
  scale?: MLOperand;
  bias?: MLOperand;
  axis?: number;
  epsilon?: number;
  activation?: MLOperator;
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-mlclampoptions)
 */
export interface MLClampOptions {
  minValue?: number;
  maxValue?: number;
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#enumdef-mlconv2dfilteroperandlayout)
 */
export enum MLConv2dFilterOperandLayout {
  'oihw' = 'oihw',
  'hwio' = 'hwio',
  'ohwi' = 'ohwi',
  'ihwo' = 'ihwo'
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#enumdef-mlautopad)
 */
export enum MLAutoPad {
  'explicit' = 'explicit',
  'same-upper' = 'same-upper',
  'same-lower' = 'same-lower',
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-mlconv2doptions)
 */
export interface MLConv2dOptions {
  padding?: [number, number, number, number];
  strides?: [number, number];
  dilations?: [number, number];
  autoPad?: MLAutoPad;
  groups?: number;
  inputLayout?: MLInputOperandLayout;
  filterLayout?: MLConv2dFilterOperandLayout;
  bias?: MLOperand;
  activation?: MLOperator;
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#enumdef-mlconvtranspose2dfilteroperandlayout)
 */
export enum MLConvTranspose2dFilterOperandLayout {
  'iohw' = 'iohw',
  'hwoi' = 'hwoi',
  'ohwi' = 'ohwi'
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-mlconvtranspose2doptions)
 */
export interface MLConvTranspose2dOptions {
  padding?: [number, number, number, number];
  strides?: [number, number];
  dilations?: [number, number];
  outputPadding?: [number, number];
  outputSizes?: [number, number];
  autoPad?: MLAutoPad;
  groups?: number;
  inputLayout?: MLInputOperandLayout;
  filterLayout?: MLConvTranspose2dFilterOperandLayout;
  bias?: MLOperand;
  activation?: MLOperator;
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-mlgemmoptions)
 */
export interface MLGemmOptions {
  c?: MLOperand;
  alpha?: number;
  beta?: number;
  aTranspose?: boolean;
  bTranspose?: boolean;
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#enumdef-mlrecurrentnetworkweightlayout)
 */
export enum MLRecurrentNetworkWeightLayout {
  'zrn' = 'zrn',
  'rzn' = 'rzn',
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#enumdef-mlrecurrentnetworkdirection)
 */
export enum MLRecurrentNetworkDirection {
  'forward' = 'forward',
  'backward' = 'backward',
  'both' = 'both',
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-mlgruoptions)
 */
export interface MLGruOptions {
  bias?: MLOperand;
  recurrentBias?: MLOperand;
  initialHiddenState?: MLOperand;
  resetAfter?: boolean;
  returnSequence?: boolean;
  direction?: MLRecurrentNetworkDirection;
  layout?: MLRecurrentNetworkWeightLayout;
  activations?: MLOperator[];
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-mlgrucelloptions)
 */
export interface MLGruCellOptions {
  bias?: MLOperand;
  recurrentBias?: MLOperand;
  resetAfter?: boolean;
  layout?: MLRecurrentNetworkWeightLayout;
  activations?: MLOperator[];
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-mlinstancenormalizationoptions)
 */
export interface MLInstanceNormalizationOptions {
  scale?: MLOperand;
  bias?: MLOperand;
  epsilon?: number;
  layout?: MLInputOperandLayout;
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-mlleakyreluoptions)
 */
export interface MLLeakyReluOptions {
  alpha?: number;
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#enumdef-mlpaddingmode)
 */
export enum MLPaddingMode {
  'constant' = 'constant',
  'edge' = 'edge',
  'reflection' = 'reflection',
  'symmetric' = 'symmetric'
}

/**
 * [spec](hhttps://webmachinelearning.github.io/webnn/#dictdef-mlpadoptions)
 */
export interface MLPadOptions {
  mode?: MLPaddingMode;
  value?: number;
}


/**
 * [spec](https://webmachinelearning.github.io/webnn/#enumdef-mlroundingtype)
 */
 export enum  MLRoundingType {
  'floor' = 'floor',
  'ceil' = 'ceil'
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-mlpool2doptions)
 */
export interface MLPooling2dOptions {
  windowDimensions?: [number, number];
  padding?: [number, number, number, number];
  strides?: [number, number];
  dilations?: [number, number];
  autoPad?: MLAutoPad;
  layout?: MLInputOperandLayout;
  roundingType?: MLRoundingType;
  outputSizes?: [number, number];
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-mlreduceoptions)
 */
export interface MLReduceOptions {
  axes?: number[];
  keepDimensions?: boolean;
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#enumdef-mlinterpolationmode)
 */
export enum MLInterpolationMode {
  'nearest-neighbor' = 'nearest-neighbor',
  'linear' = 'linear'
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-mlresample2doptions)
 */
export interface MLResample2dOptions {
  mode?: MLInterpolationMode;
  scales?: [number, number];
  sizes?: [number, number];
  axes?: [number, number];
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-mlsliceoptions)
 */
export interface MLSliceOptions {
  axes?: number[];
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-mlsplitoptions)
 */
export interface MLSplitOptions {
  axis?: number;
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-mlsqueezeoptions)
 */
export interface MLSqueezeOptions {
  axes?: number[];
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-mltransposeoptions)
 */
export interface MLTransposeOptions {
  permutation?: number[];
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#typedefdef-mlnamedoperands)
 */
export type MLNamedOperands = Record<string, MLOperand>;

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-mlbufferresourceview)
 */
export interface MLBufferResourceView {
  resource: WebGLBuffer;
  offset?: number;
  size?: number;
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#typedefdef-mlbufferview)
 */
export type MLBufferView = ArrayBufferView|MLBufferResourceView;

/**
 * [spec](hhttps://webmachinelearning.github.io/webnn/#api-mlgraphbuilder)
 */
export class MLGraphBuilder {
  private context_: MLContext;

  constructor(context: MLContext) {
    utils.assert(
        context instanceof MLContext, 'The context paramter is invalid.');
    this.context_ = context;
  }

  /** @internal */
  get context(): MLContext {
    return this.context_;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-build)
   */
  build(outputs: MLNamedOperands): MLGraph {
    return MLGraph.buildAndCompile(outputs);
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-input)
   */
  input(name: string, desc: MLOperandDescriptor): MLOperand {
    return new InputOperand(name, desc, this);
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-constant)
   */
  constant(desc: MLOperandDescriptor, bufferView: MLBufferView): MLOperand;
  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-constant-value-type)
   */
  constant(value: number, type?: MLOperandType): MLOperand;
  constant(
      descOrValue: MLOperandDescriptor|number,
      valueOrType: MLBufferView|MLOperandType): ConstantOperand {
    if (typeof descOrValue === 'number') {
      if (valueOrType === undefined) {
        valueOrType = MLOperandType.float32;
      }
      return ConstantOperand.createScalar(
          descOrValue, valueOrType as MLOperandType, this);
    } else {
      return ConstantOperand.createTensor(
          descOrValue, valueOrType as MLBufferView, this);
    }
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-batchnormalization)
   */
  batchNormalization(
      input: MLOperand, mean: MLOperand, variance: MLOperand,
      options: MLBatchNormalizationOptions = {}): MLOperand {
    this.validateOperandBuilder(
        [input, mean, variance, options.scale, options.bias]);
    return (new BatchNormalization(input, mean, variance, options))
        .getFusedOutputs()[0];
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-clamp)
   */
  clamp(x: MLOperand, options: MLClampOptions): MLOperand;
  clamp(options: MLClampOptions): MLOperator;
  clamp(
      operandOrOptions: MLOperand|MLClampOptions = {},
      options: MLClampOptions = {}): MLOperand|MLOperator {
    if (operandOrOptions instanceof MLOperand) {
      const x = operandOrOptions;
      this.validateOperandBuilder([x]);
      return (new Clamp(x, options)).output;
    } else {
      const options = operandOrOptions;
      return (new Clamp(undefined, options));
    }
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-concat)
   */
  concat(inputs: MLOperand[], axis: number): MLOperand {
    this.validateOperandBuilder(inputs);
    return (new Concat(inputs, axis)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-conv2d)
   */
  conv2d(input: MLOperand, filter: MLOperand, options: MLConv2dOptions = {}):
      MLOperand {
    const inputs = [input, filter];
    if (options.bias) {
      inputs.push(options.bias);
    }
    this.validateOperandBuilder(inputs);
    return (new Conv2d(input, filter, options)).getFusedOutputs()[0];
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-convtranspose2d)
   */
  convTranspose2d(
      input: MLOperand, filter: MLOperand, 
      options: MLConvTranspose2dOptions = {}):MLOperand {
    const inputs = [input, filter];
    if (options.bias) {
      inputs.push(options.bias);
    }
    this.validateOperandBuilder(inputs);
    return (new ConvTranspose2d(input, filter, options)).getFusedOutputs()[0];
  }

  // start of element-wise binary operations
  // https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-binary
  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-add)
   */
  add(a: MLOperand, b: MLOperand): MLOperand {
    this.validateOperandBuilder([a, b]);
    return (new Add(a, b)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-sub)
   */
  sub(a: MLOperand, b: MLOperand): MLOperand {
    this.validateOperandBuilder([a, b]);
    return (new Sub(a, b)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-mul)
   */
  mul(a: MLOperand, b: MLOperand): MLOperand {
    this.validateOperandBuilder([a, b]);
    return (new Mul(a, b)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-div)
   */
  div(a: MLOperand, b: MLOperand): MLOperand {
    this.validateOperandBuilder([a, b]);
    return (new Div(a, b)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-max)
   */
  max(a: MLOperand, b: MLOperand): MLOperand {
    this.validateOperandBuilder([a, b]);
    return (new Max(a, b)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-min)
   */
  min(a: MLOperand, b: MLOperand): MLOperand {
    this.validateOperandBuilder([a, b]);
    return (new Min(a, b)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-pow)
   *
   */
  pow(a: MLOperand, b: MLOperand): MLOperand {
    this.validateOperandBuilder([a, b]);
    return (new Pow(a, b)).output;
  }
  // end of element-wise binary operations

  // start of element-wise unary operations
  // https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-unary
  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-abs)
   */
  abs(x: MLOperand): MLOperand {
    this.validateOperandBuilder([x]);
    return (new Abs(x)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-ceil)
   */
  ceil(x: MLOperand): MLOperand {
    this.validateOperandBuilder([x]);
    return (new Ceil(x)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-cos)
   */
  cos(x: MLOperand): MLOperand {
    this.validateOperandBuilder([x]);
    return (new Cos(x)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-exp)
   */
  exp(x: MLOperand): MLOperand {
    this.validateOperandBuilder([x]);
    return (new Exp(x)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-floor)
   */
  floor(x: MLOperand): MLOperand {
    this.validateOperandBuilder([x]);
    return (new Floor(x)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-log)
   */
  log(x: MLOperand): MLOperand {
    this.validateOperandBuilder([x]);
    return (new Log(x)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-neg)
   */
  neg(x: MLOperand): MLOperand {
    this.validateOperandBuilder([x]);
    return (new Neg(x)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-sin)
   */
  sin(x: MLOperand): MLOperand {
    this.validateOperandBuilder([x]);
    return (new Sin(x)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-tan)
   */
  tan(x: MLOperand): MLOperand {
    this.validateOperandBuilder([x]);
    return (new Tan(x)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-hard-swish)
   */
  hardSwish(input: MLOperand): MLOperand;
  hardSwish(): MLOperator;
  hardSwish(input: MLOperand = undefined): MLOperand|MLOperator {
    if (input === undefined) {
      return new HardSwish(undefined);
    } else {
      this.validateOperandBuilder([input]);
      return (new HardSwish(input)).output;
    }
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-relu)
   */
  relu(input: MLOperand): MLOperand;
  relu(): MLOperator;
  relu(input: MLOperand = undefined): MLOperand|MLOperator {
    if (input === undefined) {
      return new Relu(undefined);
    } else {
      this.validateOperandBuilder([input]);
      return (new Relu(input)).output;
    }
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-sigmoid)
   */
  sigmoid(input: MLOperand): MLOperand;
  sigmoid(): MLOperator;
  sigmoid(input: MLOperand = undefined): MLOperand|MLOperator {
    if (input === undefined) {
      return new Sigmoid(undefined);
    } else {
      this.validateOperandBuilder([input]);
      return (new Sigmoid(input)).output;
    }
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-tanh)
   */
  tanh(input: MLOperand): MLOperand;
  tanh(): MLOperator;
  tanh(input: MLOperand = undefined): MLOperand|MLOperator {
    if (input === undefined) {
      return new Tanh(undefined);
    } else {
      this.validateOperandBuilder([input]);
      return (new Tanh(input)).output;
    }
  }
  // end of element-wise unary operations

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-gemm)
   */
  gemm(a: MLOperand, b: MLOperand, options: MLGemmOptions = {}): MLOperand {
    this.validateOperandBuilder([a, b, options.c]);
    return Gemm.build(this, a, b, options);
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-gru)
   */
  gru(input: MLOperand, weight: MLOperand, recurrentWeight: MLOperand,
      steps: number, hiddenSize: number,
      options: MLGruOptions = {}): MLOperand[] {
    this.validateOperandBuilder([
      input, weight, recurrentWeight, options.bias, options.recurrentBias,
      options.initialHiddenState
    ]);
    return (new Gru(input, weight, recurrentWeight, steps, hiddenSize, options))
        .outputs;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-grucell)
   */
  gruCell(
      input: MLOperand, weight: MLOperand, recurrentWeight: MLOperand,
      hiddenState: MLOperand, hiddenSize: number,
      options: MLGruCellOptions = {}): MLOperand {
    this.validateOperandBuilder([
      input, weight, recurrentWeight, hiddenState, options.bias,
      options.recurrentBias
    ]);
    return (new GruCell(
                input, weight, recurrentWeight, hiddenState, hiddenSize,
                options))
        .output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-instancenorm)
   */
  instanceNormalization(
      input: MLOperand,
      options: MLInstanceNormalizationOptions = {}): MLOperand {
    this.validateOperandBuilder([input, options.bias, options.scale]);
    return (new InstanceNormalization(input, options)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-leakyrelu)
   */
  leakyRelu(x: MLOperand, options: MLLeakyReluOptions): MLOperand;
  leakyRelu(options: MLLeakyReluOptions): MLOperator;
  leakyRelu(
      operandOrOptions: MLOperand|MLLeakyReluOptions = {},
      options: MLLeakyReluOptions = {}): MLOperand|MLOperator {
    if (operandOrOptions instanceof MLOperand) {
      const x = operandOrOptions;
      this.validateOperandBuilder([x]);
      return (new LeakyRelu(x, options.alpha)).output;
    } else {
      const options = operandOrOptions;
      return (new LeakyRelu(undefined, options.alpha));
    }
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-matmul)
   */
  matmul(a: MLOperand, b: MLOperand): MLOperand {
    this.validateOperandBuilder([a, b]);
    return (new MatMul(a, b)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-pad)
   */
  pad(input: MLOperand, padding: MLOperand,
      options: MLPadOptions = {}): MLOperand {
    this.validateOperandBuilder([input, padding]);
    return (new Pad(input, padding, options)).output;
  }

  // start of pooling operations
  // https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-pool2d
  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-averagepool2d)
   */
  averagePool2d(input: MLOperand, options: MLPooling2dOptions = {}): MLOperand {
    this.validateOperandBuilder([input]);
    return (new AveragePool2d(input, options)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-l2pool2d)
   */
  l2Pool2d(input: MLOperand, options: MLPooling2dOptions = {}): MLOperand {
    this.validateOperandBuilder([input]);
    return (new L2Pool2d(input, options)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-maxpool2d)
   */
  maxPool2d(input: MLOperand, options: MLPooling2dOptions = {}): MLOperand {
    this.validateOperandBuilder([input]);
    return (new MaxPool2d(input, options)).output;
  }
  // end of pooling operations

  // start of reduction operations
  // https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-reduce
  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-reducel1)
   */
  reduceL1(input: MLOperand, options: MLReduceOptions = {}): MLOperand {
    this.validateOperandBuilder([input]);
    return (new ReduceL1(input, options)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-reducel2)
   */
  reduceL2(input: MLOperand, options: MLReduceOptions = {}): MLOperand {
    this.validateOperandBuilder([input]);
    return (new ReduceL2(input, options)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-reducelogsumexp)
   */
  reduceLogSumExp(input: MLOperand, options: MLReduceOptions = {}): MLOperand {
    this.validateOperandBuilder([input]);
    return (new ReduceLogSumExp(input, options)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-reducemax)
   */
  reduceMax(input: MLOperand, options: MLReduceOptions = {}): MLOperand {
    this.validateOperandBuilder([input]);
    return (new ReduceMax(input, options)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-reducemean)
   */
  reduceMean(input: MLOperand, options: MLReduceOptions = {}): MLOperand {
    this.validateOperandBuilder([input]);
    return (new ReduceMean(input, options)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-reducemin)
   */
  reduceMin(input: MLOperand, options: MLReduceOptions = {}): MLOperand {
    this.validateOperandBuilder([input]);
    return (new ReduceMin(input, options)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-reduceproduct)
   */
  reduceProduct(input: MLOperand, options: MLReduceOptions = {}): MLOperand {
    this.validateOperandBuilder([input]);
    return (new ReduceProduct(input, options)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-reducesum)
   */
  reduceSum(input: MLOperand, options: MLReduceOptions = {}): MLOperand {
    this.validateOperandBuilder([input]);
    return (new ReduceSum(input, options)).output;
  }
  // end of reduction operations

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-resample2d)
   */
  resample2d(input: MLOperand, options: MLResample2dOptions = {}): MLOperand {
    this.validateOperandBuilder([input]);
    return (new Resample2d(input, options)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-reshape)
   */
  reshape(input: MLOperand, newShape: number[]): MLOperand {
    this.validateOperandBuilder([input]);
    return (new Reshape(input, newShape)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-slice)
   */
  slice(
      input: MLOperand, starts: number[], sizes: number[],
      options: MLSliceOptions = {}): MLOperand {
    this.validateOperandBuilder([input]);
    return (new Slice(input, starts, sizes, options.axes)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-softmax)
   */
  softmax(x: MLOperand): MLOperand {
    this.validateOperandBuilder([x]);
    return (new Softmax(x)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-split)
   */
  split(
      input: MLOperand, splits: number|number[],
      options: MLSplitOptions = {}): MLOperand[] {
    this.validateOperandBuilder([input]);
    return (new Split(input, splits, options)).outputs;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-squeeze)
   */
  squeeze(input: MLOperand, options: MLSqueezeOptions = {}): MLOperand {
    this.validateOperandBuilder([input]);
    return (new Squeeze(input, options.axes)).output;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-transpose)
   */
  transpose(input: MLOperand, options: MLTransposeOptions = {}): MLOperand {
    this.validateOperandBuilder([input]);
    return (new Transpose(input, options.permutation)).output;
  }

  private validateOperandBuilder(operands: MLOperand[]) {
    utils.assert(
        operands.every(
            operand => operand ?
                (operand instanceof MLOperand && operand.builder === this) :
                true),
        'The operand is not built by this builder.');
  }
}
