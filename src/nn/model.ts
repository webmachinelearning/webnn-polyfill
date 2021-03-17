import {Compilation} from './compilation';
import {NamedOperands} from './model_builder';
import {ConstantOperand, InputOperand, OutputOperand} from './operand';
import {Operation} from './operation';
import * as utils from './utils';

/**
 * [API
 * spec](https://webmachinelearning.github.io/webnn/#enumdef-powerpreference)
 */
export enum PowerPreference {
  'default' = 'default',
  'high-performance' = 'high-performance',
  'low-power' = 'low-power'
}

/**
 * [API
 * spec](https://webmachinelearning.github.io/webnn/#dictdef-compilationoptions)
 */
export interface CompilationOptions {
  /** */
  powerPreference?: PowerPreference;
}

/**
 * [API spec](https://webmachinelearning.github.io/webnn/#model)
 */
export class Model {
  private inputs_: Map<string, InputOperand> = new Map();
  private outputs_: Map<string, OutputOperand> = new Map();
  private constants_: Set<ConstantOperand> = new Set();

  /** @ignore */
  get inputs(): Map<string, InputOperand> {
    return this.inputs_;
  }
  /** @ignore */
  get outputs(): Map<string, OutputOperand> {
    return this.outputs_;
  }
  /** @ignore */
  get constants(): ConstantOperand[] {
    return Array.from(this.constants_.values());
  }

  /** @ignore */
  constructor(outputs?: NamedOperands) {
    utils.assert(outputs !== undefined, 'Invalid argument');
    for (const name in outputs) {
      utils.assert(
          typeof name === 'string' && outputs[name] instanceof OutputOperand,
          'The outputs parameter is invalid.');
      this.outputs_.set(name, outputs[name] as OutputOperand);
    }
    utils.assert(this.outputs_.size !== 0, 'The outputs is empty');
    this.initialize();
  }

  async compile(options: CompilationOptions): Promise<Compilation> {
    const compilation = await Compilation.createAndCompile(options, this);
    return compilation;
  }

  private initialize(): void {
    const visitedOps: Set<Operation> = new Set();
    for (const output of this.outputs_.values()) {
      this.handleOperation(output.operation, visitedOps);
    }
  }

  private handleOperation(operation: Operation, visitedOps: Set<Operation>):
      void {
    if (visitedOps.has(operation)) {
      return;
    } else {
      visitedOps.add(operation);
    }
    for (const operand of operation.inputs()) {
      if (operand instanceof InputOperand) {
        if (this.inputs_.has(operand.name)) {
          if (this.inputs_.get(operand.name) !== operand) {
            throw new Error('The name of this input is existed.');
          } else {
            continue;
          }
        }
        this.inputs_.set(operand.name, operand);
      } else if (operand instanceof ConstantOperand) {
        if (!this.constants_.has(operand)) {
          this.constants_.add(operand);
        }
      } else if (operand instanceof OutputOperand) {
        this.handleOperation(operand.operation, visitedOps);
      }
    }
  }
}
