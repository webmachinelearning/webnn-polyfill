import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';

import * as tf from '@tensorflow/tfjs-core';

import {Compilation as CompilationInterface} from './compilation';
import {CompilationOptions} from './compilation_options';
import {Constant} from './constant';
import {ExecutionContext} from './execution_context';
import {Execution} from './execution_impl';
import {Input} from './input';
import {Model} from './model_impl';
import {OperandDescriptor} from './operand_descriptor';
import {Output} from './output';
import * as utils from './utils';

export class Compilation implements CompilationInterface {
  private model_: Model;
  private constantTensors_: Map<Constant, tf.Tensor> = new Map();
  private outputDescriptors_: Map<Output, OperandDescriptor> = new Map();

  get model(): Model {
    return this.model_;
  }
  get constantTensors(): Map<Constant, tf.Tensor> {
    return this.constantTensors_;
  }
  get outputDescriptors(): Map<Output, OperandDescriptor> {
    return this.outputDescriptors_;
  }

  /** */
  async createExecution(): Promise<Execution> {
    return new Execution(this);
  }

  static async createAndCompile(options: CompilationOptions, model: Model):
      Promise<Compilation> {
    const compilation = new Compilation(options, model);
    await compilation.compile();
    return compilation;
  }

  constructor(options?: CompilationOptions, model?: Model) {
    utils.assert(typeof model !== 'undefined', 'Invalid arguments');
    // TODO: support compilation options.
    this.model_ = model;
  }

  private async compile(): Promise<void> {
    try {
      if (!(await tf.setBackend('webgl'))) {
        console.warn(
            'Failed to set tf.js webgl backend, fallback to cpu backend.');
        if (!(await tf.setBackend('cpu'))) {
          throw new Error('Failed to set tf.js cpu backend.');
        }
      }
    } catch (error) {
      // webgl backend is not registered for node.js
      if (!(await tf.setBackend('cpu'))) {
        throw new Error('Failed to set tf.js cpu backend.');
      }
    }
    await tf.ready();
    this.allocateConstants();
    await this.inferOnce();
  }

  private allocateConstants(): void {
    for (const constant of this.model_.constants) {
      this.constantTensors_.set(
          constant, utils.createTensor(constant.desc, constant.value));
    }
  }

  private async inferOnce(): Promise<void> {
    const inputTensors: Map<Input, tf.Tensor> = new Map();
    for (const input of this.model_.inputs.values()) {
      const typedArrayConstructor = utils.getTypedArray(input.desc.type);
      const inputBuffer = new typedArrayConstructor(
          utils.sizeFromDimensions(input.desc.dimensions));
      inputTensors.set(input, utils.createTensor(input.desc, inputBuffer));
    }
    for (const output of this.model_.outputs.values()) {
      const tensor: tf.Tensor = tf.tidy(
          () => output.operation.run(
              {inputTensors, constantTenosrs: this.constantTensors_} as
              ExecutionContext));
      await tensor.data();
      this.outputDescriptors_.set(
          output, utils.createOperandDescriptorFromTensor(tensor));
      tf.dispose(tensor);
    }
    for (const tensor of inputTensors.values()) {
      tf.dispose(tensor);
    }
  }
}