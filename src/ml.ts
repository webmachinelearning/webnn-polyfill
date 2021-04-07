import {MLContext, MLContextOptions} from './nn/context';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#ml)
 */
export class ML {
  /** @ignore */
  // eslint-disable-next-line @typescript-eslint/no-empty-function
  constructor() {}

  createContext(options: MLContextOptions = {}): MLContext {
    return new MLContext(options);
  }
}
