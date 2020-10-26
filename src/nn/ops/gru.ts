import {ModelBuilder} from '../model_builder_impl';
import {Operand} from '../operand_impl';
import {RecurrentNetworkActivation, RecurrentNetworkWeightLayout} from '../types';
import * as utils from '../utils';

export class GruCell {
  static build(
      builder: ModelBuilder, input: Operand, weight: Operand,
      recurrentWeight: Operand, hiddenState: Operand, hiddenSize: number,
      bias?: Operand, recurrentBias?: Operand, resetAfter = true,
      layout: RecurrentNetworkWeightLayout = RecurrentNetworkWeightLayout.zrn,
      activations: RecurrentNetworkActivation[] = [
        RecurrentNetworkActivation.sigmoid, RecurrentNetworkActivation.tanh
      ]): Operand {
    utils.assert(
        utils.isInteger(hiddenSize),
        'The hiddenSize parameter is not an integer.');
    utils.assert(
        utils.isBoolean(resetAfter),
        'The resetAfter parameter is not a boolean.');
    utils.assert(
        layout in RecurrentNetworkWeightLayout,
        'The layout parameter is invalid.');
    utils.assert(
        activations instanceof Array && activations.length === 2 &&
            activations.every(a => a in RecurrentNetworkActivation),
        'The activations parameter is invalid.');
    const one = builder.constant(1);
    const zero = builder.constant(0);
    const starts = layout === RecurrentNetworkWeightLayout.zrn ?
        {z: 0, r: hiddenSize, n: 2 * hiddenSize} :
        /*rzn*/ {r: 0, z: hiddenSize, n: 2 * hiddenSize};
    // update gate
    const z = builder[activations[0]](builder.add(
        builder.add(
            (bias ? builder.slice(bias, [starts.z], [hiddenSize]) : zero),
            (recurrentBias ?
                 builder.slice(recurrentBias, [starts.z], [hiddenSize]) :
                 zero)),
        builder.add(
            builder.matmul(
                input,
                builder.transpose(
                    builder.slice(weight, [starts.z, 0], [hiddenSize, -1]))),
            builder.matmul(
                hiddenState,
                builder.transpose(builder.slice(
                    recurrentWeight, [starts.z, 0], [hiddenSize, -1]))))));
    // reset gate
    const r = builder[activations[0]](builder.add(
        builder.add(
            (bias ? builder.slice(bias, [starts.r], [hiddenSize]) : zero),
            (recurrentBias ?
                 builder.slice(recurrentBias, [starts.r], [hiddenSize]) :
                 zero)),
        builder.add(
            builder.matmul(
                input,
                builder.transpose(
                    builder.slice(weight, [starts.r, 0], [hiddenSize, -1]))),
            builder.matmul(
                hiddenState,
                builder.transpose(builder.slice(
                    recurrentWeight, [starts.r, 0], [hiddenSize, -1]))))));
    // new gate
    let n;
    if (resetAfter) {
      n = builder[activations[1]](builder.add(
          (bias ? builder.slice(bias, [starts.n], [hiddenSize]) : zero),
          builder.add(
              builder.matmul(
                  input,
                  builder.transpose(
                      builder.slice(weight, [starts.n, 0], [hiddenSize, -1]))),
              builder.mul(
                  r,
                  builder.add(
                      (recurrentBias ?
                           builder.slice(
                               recurrentBias, [starts.n], [hiddenSize]) :
                           zero),
                      builder.matmul(
                          hiddenState,
                          builder.transpose(builder.slice(
                              recurrentWeight, [starts.n, 0],
                              [hiddenSize, -1]))))))));
    } else {
      n = builder[activations[1]](builder.add(
          builder.add(
              (bias ? builder.slice(bias, [starts.n], [hiddenSize]) : zero),
              (recurrentBias ?
                   builder.slice(recurrentBias, [starts.n], [hiddenSize]) :
                   zero)),
          builder.add(
              builder.matmul(
                  input,
                  builder.transpose(
                      builder.slice(weight, [starts.n, 0], [hiddenSize, -1]))),
              builder.matmul(
                  builder.mul(r, hiddenState),
                  builder.transpose(builder.slice(
                      recurrentWeight, [starts.n, 0], [hiddenSize, -1]))))));
    }
    // compute the new hidden state
    return builder.add(
        builder.mul(z, hiddenState), builder.mul(n, builder.sub(one, z)));
  }
}
