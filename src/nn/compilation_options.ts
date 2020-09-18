import {PowerPreference} from './power_preference';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-compilationoptions)
 */
export interface CompilationOptions {
  /** */
  powerPreference?: PowerPreference;
}