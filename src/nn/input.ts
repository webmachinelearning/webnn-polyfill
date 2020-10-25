import {ArrayBufferView} from './types';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-input)
 */
export interface Input {
  /** */
  readonly buffer: ArrayBufferView;
  /** */
  readonly dimensions?: number[];
}