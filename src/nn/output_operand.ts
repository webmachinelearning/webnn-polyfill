import {Operand} from './operand_impl';
import {Operation} from './operation';

export class OutputOperand extends Operand {
  readonly operation: Operation;

  constructor(operation: Operation) {
    super();
    this.operation = operation;
  }
}