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

  compute(context: ExecutionContext): void {
    const inputTensors: Map<Operand, tf.Tensor> = new Map();
    for (const inputOperand of this.inputs()) {
      inputTensors.set(inputOperand, context.getTensor(inputOperand));
    }
    const outputTensors = tf.tidy(() => this.computeImpl(inputTensors));
    for (let i = 0; i < this.outputs_.length; ++i) {
      context.setOutputTensor(this.outputs_[i], outputTensors[i]);
    }
    for (const inputOperand of this.inputs()) {
      context.releaseTensor(inputOperand);
    }
  }

  abstract computeImpl(inputTensors: Map<Operand, tf.Tensor>): tf.Tensor[];
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

  computeImpl(inputTensors: Map<Operand, tf.Tensor>): tf.Tensor[] {
    return [this.run(inputTensors)];
  }

  abstract run(inputTensors: Map<Operand, tf.Tensor>): tf.Tensor;
}