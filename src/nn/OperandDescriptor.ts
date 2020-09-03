import {OperandType} from './OperandType';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-operanddescriptor)
 */
export interface OperandDescriptor {
  /** */
  type: OperandType;
  /** */
  dimensions: number[];
}