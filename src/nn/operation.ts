import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from './compilation';
import {ModelBuilder} from './model_builder';
import {Operand, OutputOperand} from './operand';

export abstract class Operation {
  protected readonly builder_: ModelBuilder;
  protected outputs_: OutputOperand[] = [];

  get builder(): ModelBuilder {
    return this.builder_;
  }

  get outputs(): OutputOperand[] {
    return this.outputs_;
  }

  constructor(builder: ModelBuilder) {
    this.builder_ = builder;
  }

  abstract inputs(): Operand[];
  abstract compute(context: ExecutionContext): void;
}

export abstract class SingleOutputOperation extends Operation {
  constructor(builder: ModelBuilder) {
    super(builder);
    // Operation produces 1 output operand by default.
    this.outputs_.push(new OutputOperand(this));
  }

  get output(): OutputOperand {
    return this.outputs_[0];
  }

  compute(context: ExecutionContext): void {
    context.setOutputTensor(this.output, this.run(context));
  }

  abstract run(context: ExecutionContext): tf.Tensor;
}