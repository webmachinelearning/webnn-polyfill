import {ArrayBufferView} from './types';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-output)
 */
export interface Output {
  /** */
  readonly buffer?: ArrayBufferView;
  /** */
  readonly dimensions?: number[];
}