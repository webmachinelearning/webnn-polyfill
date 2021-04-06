import * as tf from '@tensorflow/tfjs-core';

import {ExecutionContext} from './graph';
import {MLGraphBuilder} from './graph_builder';
import {MLOperand, OutputOperand} from './operand';

export abstract class Operation {
  protected readonly builder_: MLGraphBuilder;
  protected outputs_: OutputOperand[] = [];

  get builder(): MLGraphBuilder {
    return this.builder_;
  }

  get outputs(): OutputOperand[] {
    return this.outputs_;
  }

  constructor(builder: MLGraphBuilder) {
    this.builder_ = builder;
  }

  abstract inputs(): MLOperand[];

  compute(context: ExecutionContext): void {
    const inputTensors: Map<MLOperand, tf.Tensor> = new Map();
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

  abstract computeImpl(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor[];
}

export abstract class SingleOutputOperation extends Operation {
  constructor(builder: MLGraphBuilder) {
    super(builder);
    // Operation produces 1 output operand by default.
    this.outputs_.push(new OutputOperand(this));
  }

  get output(): OutputOperand {
    return this.outputs_[0];
  }

  computeImpl(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor[] {
    return [this.run(inputTensors)];
  }

  abstract run(inputTensors: Map<MLOperand, tf.Tensor>): tf.Tensor;
}