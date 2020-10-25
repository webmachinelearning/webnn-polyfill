import {Compilation as CompilationImpl} from './compilation_impl';
import {NamedInputs} from './named_inputs';
import {NamedOutputs} from './named_outputs';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#compilation)
 */
export interface Compilation {
  /** */
  compute(inputs: NamedInputs, outputs?: NamedOutputs): Promise<NamedOutputs>;
}

interface CompilationConstructor {
  new(): Compilation;
}

// eslint-disable-next-line no-redeclare
export const Compilation: CompilationConstructor = CompilationImpl;
