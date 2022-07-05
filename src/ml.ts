import {MLContext, MLContextOptions} from './nn/context';
import * as utils from './nn/utils';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#ml)
 */
export class ML {
  /** @ignore */
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  constructor() {}

  async createContext(options: MLContextOptions = {}): Promise<MLContext> {
    return new Promise((resolve, reject) => {
      let context;
      try {
        context = new MLContext(options);
      } catch(error) {
        reject(error);
      }
      resolve(context);
    });
  }

  createContextSync(options: MLContextOptions = {}): MLContext {
    utils.assert(
        typeof window === 'undefined' && typeof importScripts === 'function',
        'createContextSync() should only be allowed in dedicated worker.');
    return new MLContext(options);
  }
}
