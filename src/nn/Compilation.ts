import {Compilation as CompilationImpl} from './compilation_impl';
import {Execution} from './execution';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#compilation)
 */
export interface Compilation {
  /** */
  createExecution(): Promise<Execution>;
}

interface ComilationConstructor {
  new(): Compilation;
}

// eslint-disable-next-line no-redeclare
export const Compilation: ComilationConstructor = CompilationImpl;
