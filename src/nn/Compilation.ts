import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';

import * as tf from '@tensorflow/tfjs-core';

import {CompilationOptions} from './CompilationOptions';
import {Constant} from './Constant';
import {Execution} from './Execution';
import {ExecutionContext} from './ExecutionContext';
import {Input} from './Input';
import {Model} from './Model';
import {OperandDescriptor} from './OperandDescriptor';
import {Output} from './Output';
import * as utils from './utils';

/**
 * Implements the
 * [Compilation](https://webmachinelearning.github.io/webnn/#compilation)
 * interface.
 */
export class Compilation {
  private model_: Model;
  private constantTensors_: Map<Constant, tf.Tensor> = new Map();
  private outputDescriptors_: Map<Output, OperandDescriptor> = new Map();

  get model() {
    return this.model_;
  }
  get constantTensors() {
    return this.constantTensors_;
  }
  get outputDescriptors() {
    return this.outputDescriptors_;
  }

  /** */
  async createExecution(): Promise<Execution> {
    return new Execution(this);
  }

  static async createAndCompile(options: CompilationOptions, model: Model):
      Promise<Compilation> {
    const compilation = new Compilation(options, model);
    await compilation.compile_();
    return compilation;
  }

  private constructor(options: CompilationOptions, model: Model) {
    // TODO: support compilation options.
    this.model_ = model;
  }

  private async compile_(): Promise<void> {
    if (!(await tf.setBackend('webgl'))) {
      console.warn(
          'Failed to set tf.js webgl backend, fallback to cpu backend.');
      if (!(await tf.setBackend('cpu'))) {
        throw new Error('Failed to set tf.js cpu backend.');
      }
    }
    await tf.ready();
    this.allocateConstants_();
    this.inferOutputShapes_();
  }

  private allocateConstants_() {
    for (const constant of this.model_.constants) {
      this.constantTensors_.set(
          constant, utils.createTensor(constant.desc, constant.value));
    }
  }

  private inferOutputShapes_() {
    const inputTensors: Map<Input, tf.Tensor> = new Map();
    for (const input of this.model_.inputs.values()) {
      const typedArrayConstructor = utils.getTypedArray(input.desc.type);
      const inputBuffer = new typedArrayConstructor(
          utils.sizeFromDimensions(input.desc.dimensions));
      inputTensors.set(input, utils.createTensor(input.desc, inputBuffer));
    }
    for (const output of this.model_.outputs.values()) {
      const tensor: tf.Tensor = tf.tidy(() => {
        return output.operation.run(
            {inputTensors, constantTenosrs: this.constantTensors_} as
            ExecutionContext);
      });
      this.outputDescriptors_.set(
          output, utils.createOperandDescriptorFromTensor(tensor));
      tf.dispose(tensor);
    }
    for (const tensor of inputTensors.values()) {
      tf.dispose(tensor);
    }
  }
}