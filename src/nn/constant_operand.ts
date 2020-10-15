import {ModelBuilder} from './model_builder_impl';
import {OperandDescriptor} from './operand_descriptor';
import {Operand} from './operand_impl';
import {OperandType} from './operand_type';
import {ArrayBufferView as TypedArray} from './types';
import * as utils from './utils';

export class ConstantOperand extends Operand {
  readonly desc: OperandDescriptor;
  readonly value: number|TypedArray;

  static createScalar(
      value: number, type: OperandType = OperandType.float32,
      builder: ModelBuilder): ConstantOperand {
    utils.assert(type in OperandType, 'The operand type is invalid.');
    utils.validateValueType(value, type);
    return new ConstantOperand({type} as OperandDescriptor, value, builder);
  }

  static createTensor(
      desc: OperandDescriptor, value: TypedArray,
      builder: ModelBuilder): ConstantOperand {
    utils.validateOperandDescriptor(desc);
    utils.validateTypedArray(value, desc.type, desc.dimensions);
    return new ConstantOperand(desc, value, builder);
  }

  private constructor(
      desc: OperandDescriptor, value: number|TypedArray,
      builder: ModelBuilder) {
    super(builder);
    this.desc = desc;
    this.value = value;
  }
}