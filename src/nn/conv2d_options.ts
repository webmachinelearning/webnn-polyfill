import {OperandLayout} from './operand_layout';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-conv2doptions)
 */
export interface Conv2dOptions {
  padding?: [number, number, number, number];
  strides?: [number, number];
  dilations?: [number, number];
  groups?: number;
  layout?: OperandLayout;
}