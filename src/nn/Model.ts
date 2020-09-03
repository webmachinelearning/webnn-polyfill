import {Compilation} from './Compilation';
import {CompilationOptions} from './CompilationOptions';
import {Model as ModelImpl} from './ModelImpl';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#model)
 */
export interface Model {
  /** */
  createCompilation(options: CompilationOptions): Promise<Compilation>;
}

interface ModelConstructor {
  new(): Model;
}
// eslint-disable-next-line no-redeclare
export const Model: ModelConstructor = ModelImpl;
