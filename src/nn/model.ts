import {Compilation} from './compilation_impl';
import {CompilationOptions} from './compilation_options';
import {Model as ModelImpl} from './model_impl';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#model)
 */
export interface Model {
  /** */
  compile(options: CompilationOptions): Promise<Compilation>;
}

interface ModelConstructor {
  new(): Model;
}
// eslint-disable-next-line no-redeclare
export const Model: ModelConstructor = ModelImpl;
