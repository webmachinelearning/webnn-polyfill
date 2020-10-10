import {OperandDescriptor} from './operand_descriptor';
import {Operand} from './operand_impl';
import {OperandType} from './operand_type';
import {ArrayBufferView as TypedArray} from './types';
import * as utils from './utils';

export class ConstantOperand extends Operand {
  readonly desc: OperandDescriptor;
  readonly value: number|TypedArray;

  static createScalar(value: number, type: OperandType = OperandType.float32):
      ConstantOperand {
    utils.assert(type in OperandType, 'The operand type is invalid.');
    utils.validateValueType(value, type);
    return new ConstantOperand({type} as OperandDescriptor, value);
  }

  static createTensor(desc: OperandDescriptor, value: TypedArray):
      ConstantOperand {
    utils.validateOperandDescriptor(desc);
    utils.validateTypedArray(value, desc);
    return new ConstantOperand(desc, value);
  }

  private constructor(desc: OperandDescriptor, value: number|TypedArray) {
    super();
    this.desc = desc;
    this.value = value;
  }
}