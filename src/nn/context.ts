import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-cpu';
import * as wasm from '@tensorflow/tfjs-backend-wasm';

import * as tf from '@tensorflow/tfjs-core';

import * as utils from './utils';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#enumdef-mlpowerpreference)
 */
export enum MLPowerPreference {
  'default' = 'default',
  'high-performance' = 'high-performance',
  'low-power' = 'low-power'
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#enumdef-mldevicepreference)
 */
export enum MLDevicePreference {
  'default' = 'default',
  'gpu' = 'gpu',
  'cpu' = 'cpu'
}


/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-mlcontextoptions)
 */
export interface MLContextOptions {
  /** */
  powerPreference?: MLPowerPreference;
  /** */
  devicePreference?: MLDevicePreference;
}

/**
 * [API spec](https://webmachinelearning.github.io/webnn/#mlcontext)
 */
export class MLContext {
  private options_: MLContextOptions;

  /** @internal */
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  constructor(options: MLContextOptions = {}) {
    utils.assert(options instanceof Object, 'Invalid options.');
    if (options.powerPreference !== undefined) {
      utils.assert(
          options.powerPreference in MLPowerPreference,
          'Invalid power preference.');
    }
    this.options_ = options;
  }

  /** @internal */
  get options(): MLContextOptions {
    return this.options_;
  }

  /** @internal */
  // Expose tf.js for backend debugging.
  get tf(): unknown {
    // Set directory of wasm binaries for 'wasm' backend
    wasm.setWasmPaths(
        `https://unpkg.com/@tensorflow/tfjs-backend-wasm@${tf.version_core}/dist/`);
    return tf;
  }
}
