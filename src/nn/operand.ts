import {ModelBuilder} from './model_builder';
import {Operand as OperandImpl} from './operand_impl';


/**
 * [spec](https://webmachinelearning.github.io/webnn/#operand)
 */
// eslint-disable-next-line @typescript-eslint/no-empty-interface
export interface Operand {}

interface OperandConstructor {
  new(builder: ModelBuilder): Operand;
}

// eslint-disable-next-line no-redeclare
export const Operand: OperandConstructor = OperandImpl;
