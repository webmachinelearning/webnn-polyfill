import {OperandLayout} from './operand_layout';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-pool2doptions)
 */
export interface Pooling2dOptions {
  windowDimensions?: [number, number];
  padding?: [number, number, number, number];
  strides?: [number, number];
  dilations?: [number, number];
  layout?: OperandLayout;
}
