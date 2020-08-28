import {Operand} from './Operand';

/**
 * Implements the
 * [NamedOperand](https://webmachinelearning.github.io/webnn/#dictdef-namedoperand)
 * dictionary.
 */
export interface NamedOperand {
  /** */
  name: string;
  /** */
  operand: Operand;
}