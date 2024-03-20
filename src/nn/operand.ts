import {MLGraphBuilder} from './graph_builder';
import {Operation} from './operation';
import {ArrayBufferView} from './types';
import * as utils from './utils';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#enumdef-MLOperandDataType)
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
  readonly desc: MLOperandDescriptor;

  /** @internal */
  get builder(): MLGraphBuilder {
    return this.builder_;
  }

  /** @internal */
  constructor(builder: MLGraphBuilder, desc: MLOperandDescriptor) {
    this.builder_ = builder;
    this.desc = desc;
  }

  dataType(): MLOperandDataType {
    return this.desc.dataType;
  }

  shape(): number[] {
    let resultShape: number[] = [];
    if (this.desc.dimensions) {
      resultShape = this.desc.dimensions.slice();
    }
    return resultShape;
  }

  rank(): number {
    return this.desc.dimensions.length;
  }
}

/** @internal */
export class InputOperand extends MLOperand {
  readonly name: string;

  constructor(
      name: string, desc: MLOperandDescriptor, builder: MLGraphBuilder) {
    super(builder, desc);
    utils.assert(typeof name === 'string', 'The name parameter is invalid');
    this.name = name;
    utils.validateOperandDescriptor(desc);
  }
}

/** @internal */
export class ConstantOperand extends MLOperand {
  readonly value: number|ArrayBufferView;

  static createScalar(
      value: number, type: MLOperandDataType = MLOperandDataType.float32,
      builder: MLGraphBuilder): ConstantOperand {
    utils.assert(
        type in MLOperandDataType, 'The operand data type is invalid.');
    utils.validateValueType(value, type);
    return new ConstantOperand(
        {dataType: type} as MLOperandDescriptor, value, builder);
  }

  static createTensor(
      desc: MLOperandDescriptor, value: ArrayBufferView,
      builder: MLGraphBuilder): ConstantOperand {
    utils.assert(
        utils.isTypedArray(value),
        'Only ArrayBufferView value type is supported.');
    const array = value ;
    utils.validateOperandDescriptor(desc);
    utils.validateTypedArray(array, desc.dataType, desc.dimensions);
    return new ConstantOperand(desc, array.slice(), builder);
  }

  private constructor(
      desc: MLOperandDescriptor, value: number|ArrayBufferView,
      builder: MLGraphBuilder) {
    super(builder, desc);
    this.value = value;
  }
}

/** @ignore */
export class OutputOperand extends MLOperand {
  readonly operation: Operation;

  constructor(operation: Operation, desc: MLOperandDescriptor) {
    super(operation.builder, desc);
    this.operation = operation;
  }
}
