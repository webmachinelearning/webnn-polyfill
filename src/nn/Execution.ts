import {Execution as ExecutionImpl} from './ExecutionImpl';
import {ArrayBufferView} from './types';

/**
 * [spec](https://webmachinelearning.github.io/webnn/#execution)
 */
export interface Execution {
  /** */
  setInput(name: string, data: ArrayBufferView): void;

  /** */
  setOutput(name: string, data: ArrayBufferView): void;

  /** */
  startCompute(): Promise<void>;
}

interface ExecutionConstructor {
  new(): Execution;
}

// eslint-disable-next-line no-redeclare
export const Execution: ExecutionConstructor = ExecutionImpl;
