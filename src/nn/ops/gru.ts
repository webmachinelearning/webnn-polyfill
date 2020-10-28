import {RecurrentNetworkActivation, RecurrentNetworkDirection, RecurrentNetworkWeightLayout} from '../gru_options';
import {ModelBuilder} from '../model_builder_impl';
import {Operand} from '../operand_impl';
import {OperandType} from '../operand_type';
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

export class Gru {
  static build(
      builder: ModelBuilder, input: Operand, weight: Operand,
      recurrentWeight: Operand, steps: number, hiddenSize: number,
      bias?: Operand, recurrentBias?: Operand, initialHiddenState?: Operand,
      resetAfter = true, returnSequence = false,
      direction: RecurrentNetworkDirection = RecurrentNetworkDirection.forward,
      layout: RecurrentNetworkWeightLayout = RecurrentNetworkWeightLayout.zrn,
      activations: RecurrentNetworkActivation[] = [
        RecurrentNetworkActivation.sigmoid, RecurrentNetworkActivation.tanh
      ]): Operand[] {
    utils.assert(
        utils.isInteger(steps), 'The steps parameter is not an integer.');
    utils.assert(
        utils.isInteger(hiddenSize),
        'The hiddenSize parameter is not an integer.');
    utils.assert(
        utils.isBoolean(resetAfter),
        'The resetAfter parameter is not a boolean.');
    utils.assert(
        utils.isBoolean(returnSequence),
        'The resetAfter parameter is not a boolean.');
    utils.assert(
        direction in RecurrentNetworkDirection,
        'The direction parameter is invalid.');
    utils.assert(
        layout in RecurrentNetworkWeightLayout,
        'The layout parameter is invalid.');
    utils.assert(
        activations instanceof Array && activations.length === 2 &&
            activations.every(a => a in RecurrentNetworkActivation),
        'The activations parameter is invalid.');

    const numDirections =
        (direction === RecurrentNetworkDirection.both ? 2 : 1);
    let hiddenState = initialHiddenState;

    if (hiddenState !== undefined) {
      const desc = {
        type: OperandType.float32,
        dimensions: [numDirections, 1, hiddenSize]
      };
      const totalSize = numDirections * hiddenSize;
      hiddenState = builder.constant(desc, new Float32Array(totalSize).fill(0));
    }

    let sequence: Operand = null;
    const cellWeight: Operand[] = [];
    const cellRecurrentWeight: Operand[] = [];
    const cellBias: Operand[] = [];
    const cellRecurrentBias: Operand[] = [];

    for (let slot = 0; slot < numDirections; ++slot) {
      cellWeight.push(builder.squeeze(
          builder.slice(weight, [slot, 0, 0], [1, -1, -1]), [0]));
      cellRecurrentWeight.push(builder.squeeze(
          builder.slice(recurrentWeight, [slot, 0, 0], [1, -1, -1]), [0]));
      cellBias.push(
          bias ?
              (builder.squeeze(builder.slice(bias, [slot, 0], [1, -1]), [0])) :
              null);
      cellRecurrentBias.push(
          recurrentBias ?
              (builder.squeeze(
                  builder.slice(recurrentBias, [slot, 0], [1, -1]), [0])) :
              null);
    }

    for (let step = 0; step < steps; ++step) {
      const cellHidden: Operand[] = [];
      let cellOutput: Operand = null;

      for (let slot = 0; slot < numDirections; ++slot) {
        cellHidden.push(builder.squeeze(
            builder.slice(hiddenState, [slot, 0, 0], [1, -1, -1]), [0]));
      }

      for (let slot = 0; slot < numDirections; ++slot) {
        const slice =
            (slot === 1 || direction === RecurrentNetworkDirection.backward ?
                 steps - step - 1 :
                 step);
        const cellInput = builder.squeeze(
            builder.slice(input, [slice, 0, 0], [1, -1, -1]), [0]);

        const result = builder.reshape(
            builder.gruCell(
                cellInput, cellWeight[slot], cellRecurrentWeight[slot],
                cellHidden[slot], hiddenSize, {
                  bias: cellBias[slot],
                  recurrentBias: cellRecurrentBias[slot],
                  resetAfter,
                  layout,
                  activations
                }),
            [1, -1, hiddenSize]);

        cellOutput =
            (cellOutput ? builder.concat([cellOutput, result], 0) : result);
      }

      hiddenState = cellOutput;

      if (returnSequence) {
        cellOutput =
            builder.reshape(cellOutput, [1, numDirections, -1, hiddenSize]);
        sequence =
            (sequence ? builder.concat([sequence, cellOutput], 0) : cellOutput);
      }
    }
    return (sequence ? [hiddenState, sequence] : [hiddenState]);
  }
}
