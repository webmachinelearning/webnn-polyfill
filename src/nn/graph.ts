import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-wasm';

import * as tf from '@tensorflow/tfjs-core';

import {MLBufferView, MLNamedOperands} from './graph_builder';
import {ConstantOperand, InputOperand, MLOperand, MLOperandDescriptor, OutputOperand} from './operand';
import {Operation} from './operation';
import {ArrayBufferView} from './types';
import * as utils from './utils';

export type MLResourcce = MLBufferView|WebGLTexture;

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-mlinput)
 */
export interface MLInput {
  resource: MLResourcce;
  dimensions: number[];
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#typedefdef-mlnamedinputs)
 */
export type MLNamedInputs = Record<string, MLInput|MLResourcce>;

/**
 * [spec](https://webmachinelearning.github.io/webnn/#typedefdef-mlnamedoutputs)
 */
export type MLNamedOutputs = Record<string, MLResourcce>;

/** @internal */
class OperandTensor {
  ref: number;
  tensor: tf.Tensor;
}

/** @internal */
export class ExecutionContext {
  private constantTenosrs_: Map<ConstantOperand, tf.Tensor>;
  private inputTensors_: Map<InputOperand, OperandTensor>;
  private outputTensors_: Map<OutputOperand, OperandTensor>;
  private operandRefs_: Map<MLOperand, number>;
  private outputOperands_: Set<OutputOperand>;

  constructor(
      constantTensors: Map<ConstantOperand, tf.Tensor>,
      inputOperands: Map<string, InputOperand>,
      inputs: MLNamedInputs,
      operandRefs: Map<MLOperand, number>,
  ) {
    this.constantTenosrs_ = constantTensors;
    this.operandRefs_ = operandRefs;
    this.allocateInputTensors(inputOperands, inputs);
    this.outputTensors_ = new Map();
    this.outputOperands_ = new Set();
  }

  private allocateInputTensors(
      inputOperands: Map<string, InputOperand>, inputs: MLNamedInputs) {
    this.inputTensors_ = new Map();
    for (const inputName in inputs) {
      const input = inputs[inputName];
      const inputOperand = inputOperands.get(inputName);
      let desc: MLOperandDescriptor;
      let resource;
      if ((input as MLInput).dimensions !== undefined) {
        desc = {
          type: inputOperand.desc.type,
          dimensions: (input as MLInput).dimensions
        } as MLOperandDescriptor;
        resource = (input as MLInput).resource;
      } else {
        desc = inputOperand.desc;
        resource = input;
      }
      this.inputTensors_.set(inputOperand, {
        ref: this.operandRefs_.get(inputOperand),
        tensor: utils.createTensor(desc, resource as ArrayBufferView)
      });
    }
  }

  compute(outputs: Map<string, OutputOperand>): tf.TensorContainerObject {
    for (const output of outputs.values()) {
      this.outputOperands_.add(output);
    }
    const outputTensors: tf.TensorContainerObject = {};
    for (const outputName of outputs.keys()) {
      outputTensors[outputName] = this.getTensor(outputs.get(outputName));
    }
    return outputTensors;
  }

  setOutputTensor(output: OutputOperand, tensor: tf.Tensor): void {
    utils.assert(
        !this.outputTensors_.has(output), 'MLOutput already has tensor.');
    this.outputTensors_.set(
        output, {ref: this.operandRefs_.get(output), tensor});
  }

  releaseTensor(operand: MLOperand): void {
    let operandTensorMap: Map<MLOperand, OperandTensor>;
    if (operand instanceof InputOperand) {
      operandTensorMap = this.inputTensors_;
    } else if (operand instanceof OutputOperand) {
      if (this.outputOperands_.has(operand)) {
        return;
      }
      operandTensorMap = this.outputTensors_;
    } else {
      return;
    }
    const operandTensor: OperandTensor = operandTensorMap.get(operand);
    utils.assert(operandTensor !== undefined, 'No tensor found for operand.');
    operandTensor.ref--;
    if (operandTensor.ref === 0) {
      tf.dispose(operandTensor.tensor);
      operandTensorMap.delete(operand);
    }
  }

  getTensor(operand: MLOperand): tf.Tensor {
    if (operand instanceof ConstantOperand) {
      return this.constantTenosrs_.get(operand);
    } else if (operand instanceof InputOperand) {
      return this.inputTensors_.get(operand).tensor;
    } else if (operand instanceof OutputOperand) {
      if (this.outputTensors_.has(operand)) {
        return this.outputTensors_.get(operand).tensor;
      } else {
        operand.operation.compute(this);
        utils.assert(this.outputTensors_.has(operand), 'No output is set.');
        return this.outputTensors_.get(operand).tensor;
      }
    } else {
      throw new Error('The operand is invalid.');
    }
  }
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#api-mlgraph)
 */
export class MLGraph {
  private inputs_: Map<string, InputOperand> = new Map();
  private outputs_: Map<string, OutputOperand> = new Map();
  private constants_: Set<ConstantOperand> = new Set();
  private operandRefs_: Map<MLOperand, number> = new Map();
  private constantTensors_: Map<ConstantOperand, tf.Tensor> = new Map();

  compute(inputs: MLNamedInputs, outputs: MLNamedOutputs): void {
    this.validateInputs(inputs);

    // Validate and filter the required output operands.
    utils.assert(Object.keys(outputs).length !== 0, 'The outputs is invalid.');
    const outputOperands: Map<string, OutputOperand> = new Map();
    for (const outputName in outputs) {
      utils.assert(
          typeof outputName === 'string' && this.outputs_.has(outputName),
          'The name of the output is invalid.');
      utils.assert(
          utils.isTypedArray(outputs[outputName]),
          'Only output of ArrayBufferView type is supported.');
      outputOperands.set(outputName, this.outputs_.get(outputName));
    }

    // Compute the output tensors.
    const outputTensors: tf.TensorContainerObject = tf.tidy(() => {
      const context = new ExecutionContext(
          this.constantTensors_, this.inputs_, inputs, this.operandRefs_);
      // The input and immediate tensors will be cleaned up.
      return context.compute(outputOperands);
    });

    // Setup the outputs.
    for (const outputName of Object.keys(outputTensors)) {
      const tensor = outputTensors[outputName] as tf.Tensor;
      const desc = utils.createOperandDescriptorFromTensor(tensor);
      const resource = outputs[outputName] as ArrayBufferView;
      utils.validateTypedArray(resource, desc.type, desc.dimensions);
      resource.set(tensor.dataSync());
      tf.dispose(tensor);
    }
  }

  private validateInputs(inputs: MLNamedInputs) {
    for (const name in inputs) {
      utils.assert(
          typeof name === 'string' && this.inputs_.has(name),
          'The name of the input is invalid.');
      const inputOperand = this.inputs_.get(name);
      let resource;
      let dimensions;
      if ((inputs[name] as MLInput).dimensions !== undefined) {
        const input = inputs[name] as MLInput;
        resource = input.resource;
        dimensions = input.dimensions;
        utils.assert(
            resource !== undefined, 'The resource of input is undefined.');
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
        resource = inputs[name] as MLResourcce;
        utils.assert(
            !utils.isDyanmicShape(inputOperand.desc.dimensions),
            'The input dimensions is not specified.');
        dimensions = inputOperand.desc.dimensions;
      }
      utils.assert(
          utils.isTypedArray(resource),
          'Only resource of ArrayBufferView type is supported.');
      utils.validateTypedArray(
          resource as ArrayBufferView, inputOperand.desc.type, dimensions);
    }
  }

  /** @ignore */
  constructor(outputs?: MLNamedOperands) {
    utils.assert(outputs !== undefined, 'Invalid argument');
    for (const name in outputs) {
      utils.assert(
          typeof name === 'string' && outputs[name] instanceof OutputOperand,
          'The outputs parameter is invalid.');
      this.outputs_.set(name, outputs[name] as OutputOperand);
    }
    utils.assert(this.outputs_.size !== 0, 'The outputs is empty');
  }

  /** @internal */
  static buildAndCompile(outputs?: MLNamedOperands): MLGraph {
    const graph = new MLGraph(outputs);
    graph.build();
    graph.compile();
    return graph;
  }

  private build(): void {
    const visitedOps: Set<Operation> = new Set();
    for (const output of this.outputs_.values()) {
      this.buildOperation(output.operation, visitedOps);
    }
  }

  private buildOperation(operation: Operation, visitedOps: Set<Operation>):
      void {
    if (visitedOps.has(operation)) {
      return;
    } else {
      visitedOps.add(operation);
    }
    for (const operand of operation.inputs()) {
      if (!this.operandRefs_.has(operand)) {
        this.operandRefs_.set(operand, 1);
      } else {
        let ref = this.operandRefs_.get(operand);
        ref++;
        this.operandRefs_.set(operand, ref);
      }
      if (operand instanceof InputOperand) {
        if (this.inputs_.has(operand.name)) {
          if (this.inputs_.get(operand.name) !== operand) {
            throw new Error('The name of this input is existed.');
          } else {
            continue;
          }
        }
        this.inputs_.set(operand.name, operand);
      } else if (operand instanceof ConstantOperand) {
        if (!this.constants_.has(operand)) {
          this.constants_.add(operand);
        }
      } else if (operand instanceof OutputOperand) {
        this.buildOperation(operand.operation, visitedOps);
      }
    }
  }

  private compile(): void {
    this.allocateConstants();
    this.computeOnce();
  }

  private allocateConstants(): void {
    for (const constant of this.constants_) {
      this.constantTensors_.set(
          constant, utils.createTensor(constant.desc, constant.value));
    }
  }

  private computeOnce(): void {
    const inputs: MLNamedInputs = {};
    for (const inputName of this.inputs_.keys()) {
      const inputOperand = this.inputs_.get(inputName);
      // assume 1 for negative dim value.
      const shape = inputOperand.desc.dimensions.map(x => x < 0 ? 1 : x);
      const typedArrayConstructor = utils.getTypedArray(inputOperand.desc.type);
      const inputBuffer = new typedArrayConstructor(
          utils.sizeFromDimensions(inputOperand.desc.dimensions));
      inputs[inputName] = {resource: inputBuffer, dimensions: shape} as MLInput;
    }
    const outputTensors: tf.TensorContainerObject = tf.tidy(() => {
      const context = new ExecutionContext(
          this.constantTensors_, this.inputs_, inputs, this.operandRefs_);
      // The input and immediate tensors will be cleaned up.
      return context.compute(this.outputs_);
    });
    for (const outputName of Object.keys(outputTensors)) {
      const tensor = outputTensors[outputName] as tf.Tensor;
      tensor.dataSync();
      tf.dispose(tensor);
    }
  }

  /** @ignore */
  // For memory leak testing.
  dispose(): void {
    for (const tensor of this.constantTensors_.values()) {
      tf.dispose(tensor);
    }
    const visitedOps: Set<Operation> = new Set();
    for (const output of this.outputs_.values()) {
      this.disposeOperation(output.operation, visitedOps);
    }
  }

  private disposeOperation(operation: Operation, visitedOps: Set<Operation>):
      void {
    if (visitedOps.has(operation)) {
      return;
    } else {
      operation.dispose();
      visitedOps.add(operation);
    }
    for (const operand of operation.inputs()) {
      if (operand instanceof OutputOperand) {
        this.disposeOperation(operand.operation, visitedOps);
      }
    }
  }
}
