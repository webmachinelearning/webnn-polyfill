import {Compilation} from './compilation_impl';
import {CompilationOptions} from './compilation_options';
import {ConstantOperand} from './constant_operand';
import {InputOperand} from './input_operand';
import {Model as ModelInterface} from './model';
import {NamedOperands} from './named_operands';
import {Operation} from './operation';
import {OutputOperand} from './output_operand';
import * as utils from './utils';

export class Model implements ModelInterface {
  private inputs_: Map<string, InputOperand> = new Map();
  private outputs_: Map<string, OutputOperand> = new Map();
  private constants_: ConstantOperand[] = [];

  get inputs(): Map<string, InputOperand> {
    return this.inputs_;
  }
  get outputs(): Map<string, OutputOperand> {
    return this.outputs_;
  }
  get constants(): ConstantOperand[] {
    return this.constants_;
  }

  constructor(outputs?: NamedOperands) {
    utils.assert(typeof outputs !== 'undefined', 'Invalid argument');
    for (const name in outputs) {
      utils.assert(
          typeof name === 'string' && outputs[name] instanceof OutputOperand,
          'The outputs parameter is invalid.');
      this.outputs_.set(name, outputs[name] as OutputOperand);
    }
    this.initialize();
  }

  async compile(options: CompilationOptions): Promise<Compilation> {
    const compilation = await Compilation.createAndCompile(options, this);
    return compilation;
  }

  private initialize(): void {
    for (const output of this.outputs_.values()) {
      this.handleOperation(output.operation);
    }
  }

  private handleOperation(operation: Operation): void {
    for (const operand of operation.inputs) {
      if (operand instanceof InputOperand) {
        this.inputs_.set(operand.name, operand);
      } else if (operand instanceof ConstantOperand) {
        this.constants_.push(operand);
      } else if (operand instanceof OutputOperand) {
        this.handleOperation(operand.operation);
      }
    }
  }
}