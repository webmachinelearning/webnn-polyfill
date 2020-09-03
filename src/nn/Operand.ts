import {Operand as OperandImpl} from './OperandImpl';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#operand)
 */
// eslint-disable-next-line @typescript-eslint/no-empty-interface
export interface Operand {}

interface OperandConstructor {
  new(): Operand;
}

// eslint-disable-next-line no-redeclare
export const Operand: OperandConstructor = OperandImpl;
