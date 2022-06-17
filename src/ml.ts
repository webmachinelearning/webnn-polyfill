import {MLContext, MLContextOptions} from './nn/context';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#ml)
 */
export class ML {
  /** @ignore */
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  constructor() {}

  async createContext(options: MLContextOptions = {}): Promise<MLContext> {
    return new Promise((resolve) => resolve(new MLContext(options)));
  }

  createContextSync(options: MLContextOptions = {}): MLContext {
    return new MLContext(options);
  }
}
