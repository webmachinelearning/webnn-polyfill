import {Compilation as CompilationImpl} from './CompilationImpl';
import {Execution} from './Execution';

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
