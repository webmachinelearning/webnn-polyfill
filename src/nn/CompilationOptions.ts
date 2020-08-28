import { PowerPreference } from './PowerPreference';

/**
 * Implements the [CompilationOptions](https://webmachinelearning.github.io/webnn/#dictdef-compilationoptions) dictionary.
 */
export class CompilationOptions {
  /** */
  powerPreference: PowerPreference = PowerPreference.default;
}