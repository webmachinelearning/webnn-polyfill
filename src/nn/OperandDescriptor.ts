import { OperandType } from './OperandType';

/**
 * Implements the [OperandDescriptor](https://webmachinelearning.github.io/webnn/#dictdef-operanddescriptor) dictionary.
 */
export interface OperandDescriptor {
  /** */
  type: OperandType;
  /** */
  dimensions: number[];
}