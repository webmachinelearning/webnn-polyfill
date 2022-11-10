import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-cpu';
import * as wasm from '@tensorflow/tfjs-backend-wasm';

import * as tf from '@tensorflow/tfjs-core';

import { MLGraph, MLNamedArrayBufferViews } from './graph';
import * as utils from './utils';


/** @internal */
export enum MLContextType {
  'default' = 'default',
  'webgpu' = 'webgpu'
}


/**
 * [spec](https://webmachinelearning.github.io/webnn/#enumdef-mldevicetype)
 */
export enum MLDeviceType {
  'cpu' = 'cpu',
  'gpu' = 'gpu'
}


/**
 * [spec](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference)
 */
export enum MLPowerPreference {
  'default' = 'default',
  'high-performance' = 'high-performance',
  'low-power' = 'low-power'
}


/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-mlcontextoptions)
 */
export interface MLContextOptions {
  deviceType?: MLDeviceType;
  powerPreference?: MLPowerPreference;
}

/**
 * [API spec](https://webmachinelearning.github.io/webnn/#mlcontext)
 */
export class MLContext {
  private options_: MLContextOptions;
  private type_: MLContextType;

  /** @internal */
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  constructor(options: MLContextOptions = {}) {
    utils.assert(options instanceof Object, 'Invalid options.');
    if (options.deviceType !== undefined) {
      utils.assert(
          options.deviceType in MLDeviceType,
          'Invalid device type.');
    }      
    if (options.powerPreference !== undefined) {
      utils.assert(
          options.powerPreference in MLPowerPreference,
          'Invalid power preference.');
    }
    this.options_ = options;
    this.type_ = MLContextType.default;
  }

  /** @internal */
  get options(): MLContextOptions {
    return this.options_;
  }

  /** @internal */
  get type(): MLContextType {
    return this.type_;
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlcontext-compute)
   */
  async compute(
      graph: MLGraph,
      inputs: MLNamedArrayBufferViews,
      outputs: MLNamedArrayBufferViews): Promise<void> {
    await graph.compute(inputs, outputs);
  }

  /**
   * [spec](https://webmachinelearning.github.io/webnn/#dom-mlcontext-computesync)
   */
  computeSync(
      graph: MLGraph,
      inputs: MLNamedArrayBufferViews,
      outputs: MLNamedArrayBufferViews): void {
      utils.assert(
          typeof window === 'undefined' && typeof importScripts === 'function',
          'computeSync() should only be allowed in dedicated worker.');
      graph.computeSync(inputs, outputs);
  }

  /** @internal */
  // Expose tf.js for backend debugging.
  get tf(): unknown {
    // Set directory of wasm binaries for 'wasm' backend
    wasm.setWasmPaths(`https://unpkg.com/@tensorflow/tfjs-backend-wasm@${
        tf.version_core}/dist/`);
    return tf;
  }
}
