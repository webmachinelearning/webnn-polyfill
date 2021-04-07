import {MLGemmOptions, MLGraphBuilder} from '../graph_builder';
import {MLOperand} from '../operand';
import * as utils from '../utils';

export class Gemm {
  static build(
      builder: MLGraphBuilder, a: MLOperand, b: MLOperand,
      options: MLGemmOptions = {}): MLOperand {
    utils.validateOperand(a);
    utils.validateOperand(b);
    utils.validateOptionalOperand(options.c);
    utils.assert(
        options.aTranspose === undefined || utils.isBoolean(options.aTranspose),
        'The options.aTranspose is invalid.');
    utils.assert(
        options.bTranspose === undefined || utils.isBoolean(options.bTranspose),
        'The options.bTranspose is invalid.');
    utils.assert(
        options.alpha === undefined || typeof options.alpha === 'number',
        'The options.alpha is invalid.');
    utils.assert(
        options.beta === undefined || typeof options.beta === 'number',
        'The options.beta is invalid.');

    // build graph
    if (options.aTranspose) {
      a = builder.transpose(a);
    }

    if (options.bTranspose) {
      b = builder.transpose(b);
    }

    const alpha =
        builder.constant(options.alpha === undefined ? 1.0 : options.alpha);
    const beta =
        builder.constant(options.beta === undefined ? 1.0 : options.beta);

    const ab = builder.matmul(builder.mul(alpha, a), b);
    return (options.c ? builder.add(ab, builder.mul(beta, options.c)) : ab);
  }
}