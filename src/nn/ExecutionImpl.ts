import * as tf from '@tensorflow/tfjs-core';

import {Compilation} from './CompilationImpl';
import {Execution as ExecutionInterface} from './Execution';
import {ExecutionContext} from './ExecutionContext';
import {Input} from './Input';
import {Output} from './Output';
import {ArrayBufferView as TypedArray} from './types';
import * as utils from './utils';

export class Execution implements ExecutionInterface {
  private compilation_: Compilation;
  private inputTensors_: Map<Input, tf.Tensor> = new Map();
  private outputBuffers_: Map<Output, TypedArray> = new Map();

  constructor(compilation?: Compilation) {
    utils.assert(typeof compilation !== 'undefined', 'Invalid argument');
    this.compilation_ = compilation;
  }

  setInput(name: string, data: TypedArray): void {
    utils.assert(
        typeof name === 'string' && this.compilation_.model.inputs.has(name),
        'The name parameter is invalid.');
    const input = this.compilation_.model.inputs.get(name);
    utils.validateTypedArray(data, input.desc);
    this.inputTensors_.set(input, utils.createTensor(input.desc, data));
  }

  setOutput(name: string, data: TypedArray): void {
    utils.assert(
        typeof name === 'string' && this.compilation_.model.outputs.has(name),
        'The name parameter is invalid.');
    const output = this.compilation_.model.outputs.get(name);
    const desc = this.compilation_.outputDescriptors.get(output);
    utils.validateTypedArray(data, desc);
    this.outputBuffers_.set(output, data);
  }

  async startCompute(): Promise<void> {
    for (const output of this.compilation_.model.outputs.values()) {
      const tensor: tf.Tensor = tf.tidy(() => output.operation.run({
        inputTensors: this.inputTensors_,
        constantTenosrs: this.compilation_.constantTensors
      } as ExecutionContext));
      const data = await tensor.data();
      tf.dispose(tensor);
      this.outputBuffers_.get(output).set(data);
    }
    for (const tensor of this.inputTensors_.values()) {
      tf.dispose(tensor);
    }
  }
}