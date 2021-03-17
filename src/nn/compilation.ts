import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';

import * as tf from '@tensorflow/tfjs-core';

import {CompilationOptions, Model, PowerPreference} from './model';
import {ConstantOperand, InputOperand, Operand, OperandDescriptor, OutputOperand} from './operand';
import {ArrayBufferView} from './types';
import * as utils from './utils';

/**
 * [API spec](https://webmachinelearning.github.io/webnn/#dictdef-input)
 */
export interface Input {
  buffer: ArrayBufferView;
  dimensions?: number[];
}

/**
 * [API spec](https://webmachinelearning.github.io/webnn/#dictdef-output)
 */
export interface Output {
  buffer?: ArrayBufferView;
  dimensions?: number[];
}

/**
 * [API
 * spec](https://webmachinelearning.github.io/webnn/#typedefdef-namedinputs)
 */
export type NamedInputs = Record<string, Input>;

/**
 * [API
 * spec](https://webmachinelearning.github.io/webnn/#typedefdef-namedoutputs)
 */
export type NamedOutputs = Record<string, Output>;

/** @ignore */
export class ExecutionContext {
  private inputTensors_: Map<InputOperand, tf.Tensor>;
  private constantTenosrs_: Map<ConstantOperand, tf.Tensor>;
  private outputTensors_: Map<OutputOperand, tf.Tensor>;

  constructor(
      constantTensors: Map<ConstantOperand, tf.Tensor>,
      inputOperands: Map<string, InputOperand>, inputs: NamedInputs) {
    this.constantTenosrs_ = constantTensors;
    this.allocateInputTensors(inputOperands, inputs);
    this.outputTensors_ = new Map();
  }

  private allocateInputTensors(
      inputOperands: Map<string, InputOperand>, inputs: NamedInputs) {
    this.inputTensors_ = new Map();
    for (const inputName in inputs) {
      const input = inputs[inputName];
      const inputOperand = inputOperands.get(inputName);
      let desc: OperandDescriptor;
      if (input.dimensions !== undefined) {
        desc = {type: inputOperand.desc.type, dimensions: input.dimensions} as
            OperandDescriptor;
      } else {
        desc = inputOperand.desc;
      }
      this.inputTensors_.set(
          inputOperand, utils.createTensor(desc, input.buffer));
    }
  }

  compute(outputs: Map<string, OutputOperand>): tf.TensorContainerObject {
    const outputTensors: tf.TensorContainerObject = {};
    for (const outputName of outputs.keys()) {
      outputTensors[outputName] = this.getTensor(outputs.get(outputName));
    }
    return outputTensors;
  }

  setOutputTensor(output: OutputOperand, tensor: tf.Tensor): void {
    utils.assert(
        !this.outputTensors_.has(output), 'Output already has tensor.');
    this.outputTensors_.set(output, tensor);
  }

  getTensor(operand: Operand): tf.Tensor {
    if (operand instanceof ConstantOperand) {
      return this.constantTenosrs_.get(operand);
    } else if (operand instanceof InputOperand) {
      return this.inputTensors_.get(operand);
    } else if (operand instanceof OutputOperand) {
      if (this.outputTensors_.has(operand)) {
        return this.outputTensors_.get(operand);
      } else {
        operand.operation.compute(this);
        utils.assert(this.outputTensors_.has(operand), 'No output is set.');
        return this.outputTensors_.get(operand);
      }
    } else {
      throw new Error('The operand is invalid.');
    }
  }
}

/**
 * [API spec](https://webmachinelearning.github.io/webnn/#compilation)
 */
export class Compilation {
  private inputOperands_: Map<string, InputOperand> = new Map();
  private outputOperands_: Map<string, OutputOperand> = new Map();
  private constantTensors_: Map<ConstantOperand, tf.Tensor> = new Map();

  async compute(inputs: NamedInputs, outputs: NamedOutputs = {}):
      Promise<NamedOutputs> {
    this.validateInputs(inputs);

    // Filter the required output operands.
    let outputOperands: Map<string, OutputOperand>;
    if (Object.keys(outputs).length !== 0) {
      outputOperands = new Map();
      for (const outputName in outputs) {
        utils.assert(
            typeof outputName === 'string' &&
                this.outputOperands_.has(outputName),
            'The name of the output is invalid.');
        outputOperands.set(outputName, this.outputOperands_.get(outputName));
      }
    } else {
      outputOperands = this.outputOperands_;
    }

    // Compute the output tensors.
    const outputTensors: tf.TensorContainerObject = tf.tidy(() => {
      const context = new ExecutionContext(
          this.constantTensors_, this.inputOperands_, inputs);
      // The input and immediate tensors will be cleaned up.
      return context.compute(outputOperands);
    });

    // Setup the results.
    const results: NamedOutputs = {};
    for (const outputName of Object.keys(outputTensors)) {
      const tensor = outputTensors[outputName] as tf.Tensor;
      const desc = utils.createOperandDescriptorFromTensor(tensor);
      const data = await tensor.data();
      tf.dispose(tensor);
      results[outputName] = {buffer: data, dimensions: desc.dimensions} as
          Output;
      if (outputs !== undefined && outputName in outputs &&
          outputs[outputName].buffer !== undefined) {
        const buffer = outputs[outputName].buffer;
        utils.validateTypedArray(buffer, desc.type, desc.dimensions);
        outputs[outputName].buffer.set(data);
      }
    }

    return results;
  }

  /** @ignore */
  static async createAndCompile(options: CompilationOptions, model: Model):
      Promise<Compilation> {
    const compilation = new Compilation(options);
    await compilation.compile(model);
    return compilation;
  }

  /** @ignore */
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
      let dimensions;
      if (input.dimensions !== undefined) {
        dimensions = input.dimensions;
        utils.assert(
            utils.isIntegerArray(dimensions) === true,
            'The type of the input dimensions is invalid.');
        utils.assert(
            dimensions.length === inputOperand.desc.dimensions.length,
            'The rank of the input dimensions is invalid.');
        utils.assert(
            !utils.isDyanmicShape(dimensions),
            'The value of input dimensions is negative.');
        for (let i = 0; i < inputOperand.desc.dimensions.length; ++i) {
          const d = inputOperand.desc.dimensions[i];
          if (d > 0) {
            utils.assert(
                d === dimensions[i],
                'The value of the input dimensions is invalid.');
          }
        }
      } else {
        utils.assert(
            !utils.isDyanmicShape(inputOperand.desc.dimensions),
            'The input dimensions is not specified.');
        dimensions = inputOperand.desc.dimensions;
      }
      utils.validateTypedArray(
          input.buffer, inputOperand.desc.type, dimensions);
    }
  }

  private async compile(model: Model): Promise<void> {
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
      // assume 1 for negative dim value.
      const shape = inputOperand.desc.dimensions.map(x => x < 0 ? 1 : x);
      const typedArrayConstructor = utils.getTypedArray(inputOperand.desc.type);
      const inputBuffer = new typedArrayConstructor(
          utils.sizeFromDimensions(inputOperand.desc.dimensions));
      inputs[inputName] = {buffer: inputBuffer, dimensions: shape} as Input;
    }
    await this.compute(inputs);
  }

  /** @ignore */
  // For memory leak testing.
  dispose(): void {
    for (const tensor of this.constantTensors_.values()) {
      tf.dispose(tensor);
    }
  }
}
