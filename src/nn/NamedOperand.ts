import {Operand} from './Operand';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-namedoperand)
 */
export interface NamedOperand {
  /** */
  name: string;
  /** */
  operand: Operand;
}