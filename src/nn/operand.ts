import {MLBufferView, MLGraphBuilder} from './graph_builder';
import {Operation} from './operation';
import {ArrayBufferView} from './types';
import * as utils from './utils';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#enumdef-mloperanddatatype)
 */
export enum MLOperandDataType {
  'float32' = 'float32',
  'float16' = 'float16',
  'int32' = 'int32',
  'uint32' = 'uint32',
  'int8' = 'int8',
  'uint8' = 'uint8'
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-mloperanddescriptor)
 */
export interface MLOperandDescriptor {
  dataType: MLOperandDataType;
  dimensions: number[];
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#api-mloperand)
 */
export class MLOperand {
  private readonly builder_: MLGraphBuilder;

  /** @internal */
  get builder(): MLGraphBuilder {
    return this.builder_;
  }

  /** @internal */
  constructor(builder: MLGraphBuilder) {
    this.builder_ = builder;
  }
}

/** @internal */
export class InputOperand extends MLOperand {
  readonly name: string;
  readonly desc: MLOperandDescriptor;

  constructor(
      name: string, desc: MLOperandDescriptor, builder: MLGraphBuilder) {
    super(builder);
    utils.assert(typeof name === 'string', 'The name parameter is invalid');
    this.name = name;
    utils.validateOperandDescriptor(desc);
    this.desc = desc;
  }
}

/** @internal */
export class ConstantOperand extends MLOperand {
  readonly desc: MLOperandDescriptor;
  readonly value: number|ArrayBufferView;

  static createScalar(
      value: number, dataType: MLOperandDataType = MLOperandDataType.float32,
      builder: MLGraphBuilder): ConstantOperand {
    utils.assert(
        dataType in MLOperandDataType, 'The operand data type is invalid.');
    utils.validateValueType(value, dataType);
    return new ConstantOperand(
        {dataType} as MLOperandDescriptor, value, builder);
  }

  static createTensor(
      desc: MLOperandDescriptor, value: MLBufferView,
      builder: MLGraphBuilder): ConstantOperand {
    utils.assert(
        utils.isTypedArray(value),
        'Only ArrayBufferView value type is supported.');
    const array = value as ArrayBufferView;
    utils.validateOperandDescriptor(desc);
    utils.validateTypedArray(array, desc.dataType, desc.dimensions);
    return new ConstantOperand(desc, array.slice(), builder);
  }

  private constructor(
      desc: MLOperandDescriptor, value: number|ArrayBufferView,
      builder: MLGraphBuilder) {
    super(builder);
    this.desc = desc;
    this.value = value;
  }
}

/** @ignore */
export class OutputOperand extends MLOperand {
  readonly operation: Operation;

  constructor(operation: Operation) {
    super(operation.builder);
    this.operation = operation;
  }
}
