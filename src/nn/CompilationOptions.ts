import {PowerPreference} from './PowerPreference';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#dictdef-compilationoptions)
 */
export interface CompilationOptions {
  /** */
  powerPreference?: PowerPreference;
}