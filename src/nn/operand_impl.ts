import {ModelBuilder} from './model_builder_impl';
import {Operand as OperandInterface} from './operand';

export class Operand implements OperandInterface {
  private readonly builder_: ModelBuilder;

  get builder(): ModelBuilder {
    return this.builder_;
  }

  constructor(builder: ModelBuilder) {
    this.builder_ = builder;
  }
}
