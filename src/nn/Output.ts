import {Operand} from './operand_impl';
import {Operation} from './operation';

export class Output extends Operand {
  readonly operation: Operation;

  constructor(operation: Operation) {
    super();
    this.operation = operation;
  }
}