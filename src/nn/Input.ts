import {OperandDescriptor} from './operand_descriptor';
import {Operand} from './operand_impl';
import {assert, validateOperandDescriptor} from './utils';

export class Input extends Operand {
  readonly name: string;
  readonly desc: OperandDescriptor;

  constructor(name: string, desc: OperandDescriptor) {
    super();
    assert(typeof name === 'string', 'The name parameter is invalid');
    this.name = name;
    validateOperandDescriptor(desc);
    this.desc = desc;
  }
}