import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';

import * as tf from '@tensorflow/tfjs-core';

import {Compilation as CompilationInterface} from './compilation';
import {CompilationOptions} from './compilation_options';
import {ConstantOperand} from './constant_operand';
import {ExecutionContext} from './execution_context';
import {Input} from './input';
import {InputOperand} from './input_operand';
import {Model} from './model_impl';
import {NamedInputs} from './named_inputs';
import {NamedOutputs} from './named_outputs';
import {OperandDescriptor} from './operand_descriptor';
import {Output} from './output';
import {OutputOperand} from './output_operand';
import {PowerPreference} from './power_preference';
import * as utils from './utils';

export class Compilation implements CompilationInterface {
  private inputOperands_: Map<string, InputOperand> = new Map();
  private outputOperands_: Map<string, OutputOperand> = new Map();
  private constantTensors_: Map<ConstantOperand, tf.Tensor> = new Map();

  async compute(inputs: NamedInputs, outputs?: NamedOutputs):
      Promise<NamedOutputs> {
    this.validateInputs(inputs);
    const inputTensors: Map<InputOperand, tf.Tensor> = new Map();
    for (const inputName in inputs) {
      const input = inputs[inputName];
      const inputOperand = this.inputOperands_.get(inputName);
      let desc: OperandDescriptor;
      if (input.dimensions !== undefined) {
        desc = {type: inputOperand.desc.type, dimensions: input.dimensions} as
            OperandDescriptor;
      } else {
        desc = inputOperand.desc;
      }
      inputTensors.set(inputOperand, utils.createTensor(desc, input.buffer));
    }

    const outputNames: string[] = [];
    if (outputs !== undefined) {
      for (const outputName in outputs) {
        utils.assert(
            typeof outputName === 'string' &&
                this.outputOperands_.has(outputName),
            'The name of the output is invalid.');
        outputNames.push(outputName);
      }
    } else {
      for (const outputName of this.outputOperands_.keys()) {
        outputNames.push(outputName);
      }
    }

    const results: NamedOutputs = {};
    for (const outputName of outputNames) {
      const outputOperand = this.outputOperands_.get(outputName);
      const tensor: tf.Tensor = tf.tidy(
          () => outputOperand.operation.run(
              {inputTensors, constantTenosrs: this.constantTensors_} as
              ExecutionContext));
      const desc = utils.createOperandDescriptorFromTensor(tensor);
      const data = await tensor.data();
      tf.dispose(tensor);
      results[outputName] = {buffer: data, dimensions: desc.dimensions} as
          Output;
      if (outputs !== undefined && outputName in outputs &&
          outputs[outputName].buffer !== undefined) {
        const buffer = outputs[outputName].buffer;
        utils.validateTypedArray(buffer, desc);
        outputs[outputName].buffer.set(data);
      }
    }

    for (const tensor of inputTensors.values()) {
      tf.dispose(tensor);
    }

    return results;
  }

  static async createAndCompile(options: CompilationOptions, model: Model):
      Promise<Compilation> {
    const compilation = new Compilation(options);
    await compilation.compile(model);
    return compilation;
  }

  constructor(options: CompilationOptions = {}) {
    this.validateOptions(options);
  }

  private validateOptions(options: CompilationOptions) {
    utils.assert(options instanceof Object, 'Invalid options.');
    if (options.powerPreference !== undefined) {
      utils.assert(
          options.powerPreference in PowerPreference,
          'Invalid power preference.');
    }
  }

  private validateInputs(inputs: NamedInputs) {
    for (const name in inputs) {
      utils.assert(
          typeof name === 'string' && this.inputOperands_.has(name),
          'The name of the input is invalid.');
      const input = inputs[name];
      const inputOperand = this.inputOperands_.get(name);
      utils.assert(
          input.buffer !== undefined, 'The buffer of the input is undefined.');
      utils.validateTypedArray(input.buffer, inputOperand.desc);
      if (input.dimensions !== undefined) {
        const dimensions = input.dimensions;
        utils.assert(
            utils.isIntegerArray(dimensions) === true,
            'The type of the input dimensions is invalid.');
        utils.assert(
            dimensions.length === inputOperand.desc.dimensions.length,
            'The rank of the input dimensions is invalid.');
        for (let i = 0; i < inputOperand.desc.dimensions.length; ++i) {
          const d = inputOperand.desc.dimensions[i];
          if (d !== -1) {
            utils.assert(
                d === dimensions[i],
                'The value of the input dimensions is invalid.');
          }
        }
      }
    }
  }

  private async compile(model: Model): Promise<void> {
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
    this.allocateConstants(model);
    this.inputOperands_ = model.inputs;
    this.outputOperands_ = model.outputs;
    await this.inferOnce();
  }

  private allocateConstants(model: Model): void {
    for (const constant of model.constants) {
      this.constantTensors_.set(
          constant, utils.createTensor(constant.desc, constant.value));
    }
  }

  private async inferOnce(): Promise<void> {
    const inputs: NamedInputs = {};
    for (const inputName of this.inputOperands_.keys()) {
      const inputOperand = this.inputOperands_.get(inputName);
      const typedArrayConstructor = utils.getTypedArray(inputOperand.desc.type);
      const inputBuffer = new typedArrayConstructor(
          utils.sizeFromDimensions(inputOperand.desc.dimensions));
      inputs[inputName] = {buffer: inputBuffer} as Input;
    }
    await this.compute(inputs);
  }
}