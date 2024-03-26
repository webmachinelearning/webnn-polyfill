import {MLContext, MLContextOptions} from './nn/context';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#api-ml)
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
}
