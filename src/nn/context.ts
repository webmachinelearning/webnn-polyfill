import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-webgpu';
import '@tensorflow/tfjs-backend-cpu';
import * as wasm from '@tensorflow/tfjs-backend-wasm';

import * as tf from '@tensorflow/tfjs-core';

import { MLComputeResult, MLGraph, MLNamedArrayBufferViews } from './graph';
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
 * [API spec](https://webmachinelearning.github.io/webnn/#api-mlcontext)
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
   * [spec](https://webmachinelearning.github.io/webnn/#api-mlcontext-compute)
   */
  async compute(
      graph: MLGraph,
      inputs: MLNamedArrayBufferViews,
      outputs: MLNamedArrayBufferViews): Promise<MLComputeResult> {
    const result = await graph.compute(inputs, outputs);
    return result;
  }

  /** @internal */
  // Expose tf interfance for setting backend.
  get tf(): unknown {
    return tf;
  }

  /** @internal */
  // Expose wasm interface for supporting configure threads for wasm backend.
  //     wasm.setThreadsCount(n)
  get wasm(): unknown {
    return wasm;
  }
}
