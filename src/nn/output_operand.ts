import {ModelBuilder} from './model_builder_impl';
import {Operand} from './operand_impl';
import {Operation} from './operation';

export class OutputOperand extends Operand {
  readonly operation: Operation;

  constructor(operation: Operation, builder: ModelBuilder) {
    super(builder);
    this.operation = operation;
  }
}