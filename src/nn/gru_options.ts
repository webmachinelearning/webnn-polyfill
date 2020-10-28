import {Operand} from './operand_impl';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#enumdef-recurrentnetworkweightlayout)
 */
export enum RecurrentNetworkWeightLayout {
  'zrn' = 'zrn',
  'rzn' = 'rzn',
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#enumdef-recurrentnetworkactivation)
 */
export enum RecurrentNetworkActivation {
  'relu' = 'relu',
  'sigmoid' = 'sigmoid',
  'tanh' = 'tanh',
}

/**
 * [spec](https://webmachinelearning.github.io/webnn/#enumdef-recurrentnetworkdirection)
 */
export enum RecurrentNetworkDirection {
  'forward' = 'forward',
  'backward' = 'backward',
  'both' = 'both',
}

export interface GruOptions {
  bias?: Operand;
  recurrentBias?: Operand;
  initialHiddenState?: Operand;
  resetAfter?: boolean;
  returnSequence?: boolean;
  direction?: RecurrentNetworkDirection;
  layout?: RecurrentNetworkWeightLayout;
  activations?: RecurrentNetworkActivation[];
}

export interface GruCellOptions {
  bias?: Operand;
  recurrentBias?: Operand;
  resetAfter?: boolean;
  layout?: RecurrentNetworkWeightLayout;
  activations?: RecurrentNetworkActivation[];
}