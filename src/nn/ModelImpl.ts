import {Compilation} from './CompilationImpl';
import {CompilationOptions} from './CompilationOptions';
import {Constant} from './Constant';
import {Input} from './Input';
import {Model as ModelInterface} from './Model';
import {NamedOperand} from './NamedOperand';
import {Operation} from './Operation';
import {Output} from './Output';
import * as utils from './utils';

export class Model implements ModelInterface {
  private inputs_: Map<string, Input> = new Map();
  private outputs_: Map<string, Output> = new Map();
  private constants_: Constant[] = [];

  get inputs(): Map<string, Input> {
    return this.inputs_;
  }
  get outputs(): Map<string, Output> {
    return this.outputs_;
  }
  get constants(): Constant[] {
    return this.constants_;
  }

  constructor(outputs?: NamedOperand[]) {
    utils.assert(typeof outputs !== 'undefined', 'Invalid argument');
    utils.assert(
        outputs.length !== 0,
        'The length of outputs parameter should not be 0.');
    utils.assert(
        outputs.every(
            namedOutput => typeof namedOutput.name === 'string' &&
                namedOutput.operand instanceof Output),
        'The outputs parameter is invalid.');
    for (const namedOutput of outputs) {
      this.outputs_.set(namedOutput.name, namedOutput.operand as Output);
    }
    this.initialize_();
  }

  async createCompilation(options: CompilationOptions): Promise<Compilation> {
    const compilation = await Compilation.createAndCompile(options, this);
    return compilation;
  }

  private initialize_(): void {
    for (const output of this.outputs_.values()) {
      this.handleOperation_(output.operation);
    }
  }

  private handleOperation_(operation: Operation): void {
    for (const operand of operation.inputs) {
      if (operand instanceof Input) {
        this.inputs_.set(operand.name, operand);
      } else if (operand instanceof Constant) {
        this.constants_.push(operand);
      } else if (operand instanceof Output) {
        this.handleOperation_(operand.operation);
      }
    }
  }
}