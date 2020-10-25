import {Operand} from './operand';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#typedefdef-namedoperands)
 */
export type NamedOperands = Record<string, Operand>;
