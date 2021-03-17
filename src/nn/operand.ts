import {ModelBuilder} from './model_builder';
import {Operation} from './operation';
import {ArrayBufferView} from './types';
import * as utils from './utils';

/**
 * [API spec](https://webmachinelearning.github.io/webnn/#enumdef-operandtype)
 */
export enum OperandType {
  'float32' = 'float32',
  'float16' = 'float16',
  'int32' = 'int32',
  'uint32' = 'uint32',
  'int8' = 'int8',
  'uint8' = 'uint8'
}

/**
 * [API
 * spec](https://webmachinelearning.github.io/webnn/#dictdef-operanddescriptor)
 */
export interface OperandDescriptor {
  type: OperandType;
  dimensions: number[];
}

/**
 * [API spec](https://webmachinelearning.github.io/webnn/#api-operand)
 */
export class Operand {
  private readonly builder_: ModelBuilder;

  /** @ignore */
  get builder(): ModelBuilder {
    return this.builder_;
  }

  /** @ignore */
  constructor(builder: ModelBuilder) {
    this.builder_ = builder;
  }
}

/** @ignore */
export class InputOperand extends Operand {
  readonly name: string;
  readonly desc: OperandDescriptor;

  constructor(name: string, desc: OperandDescriptor, builder: ModelBuilder) {
    super(builder);
    utils.assert(typeof name === 'string', 'The name parameter is invalid');
    this.name = name;
    utils.validateOperandDescriptor(desc);
    this.desc = desc;
  }
}

/** @ignore */
export class ConstantOperand extends Operand {
  readonly desc: OperandDescriptor;
  readonly value: number|ArrayBufferView;

  static createScalar(
      value: number, type: OperandType = OperandType.float32,
      builder: ModelBuilder): ConstantOperand {
    utils.assert(type in OperandType, 'The operand type is invalid.');
    utils.validateValueType(value, type);
    return new ConstantOperand({type} as OperandDescriptor, value, builder);
  }

  static createTensor(
      desc: OperandDescriptor, value: ArrayBufferView,
      builder: ModelBuilder): ConstantOperand {
    utils.validateOperandDescriptor(desc);
    utils.validateTypedArray(value, desc.type, desc.dimensions);
    return new ConstantOperand(desc, value, builder);
  }

  private constructor(
      desc: OperandDescriptor, value: number|ArrayBufferView,
      builder: ModelBuilder) {
    super(builder);
    this.desc = desc;
    if (typeof value === 'number') {
      this.value = value;
    } else {
      this.value = utils.cloneTypedArray(value);
    }
  }
}

/** @ignore */
export class OutputOperand extends Operand {
  readonly operation: Operation;

  constructor(operation: Operation) {
    super(operation.builder);
    this.operation = operation;
  }
}
