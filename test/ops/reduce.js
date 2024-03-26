'use strict';
import * as utils from '../utils.js';

describe('test reduce', () => {
  let context;
  before(async () => {
    context = await navigator.ml.createContext();
  });

  async function testReduce(op, options, input, expected) {
    const builder = new MLGraphBuilder(context);
    const x =
        builder.input('x', {dataType: 'float32', dimensions: input.shape});
    const y = builder['reduce' + op](x, options);
    utils.checkDataType(y.dataType(), x.dataType());
    utils.checkShape(y.shape(), expected.shape);
    const graph = await builder.build({y});
    const inputs = {'x': new Float32Array(input.values)};
    const outputs = {'y': new Float32Array(utils.sizeOfShape(expected.shape))};
    const result = await context.compute(graph, inputs, outputs);
    utils.checkValue(result.outputs.y, expected.values);
  }

  it('reduceLogSum default', async () => {
    await testReduce(
        'LogSum', {}, {
          shape: [3, 2, 2],
          values: [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
          ],
        },
        {shape: [], values: [4.189654742]});
  });

  it('reduceLogSum default axes keep dims', async () => {
    await testReduce(
        'LogSum', {keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
          ],
        },
        {shape: [1, 1, 1], values: [4.189654742]});
  });

  it('reduceLogSum axes0 do not keep dims', async () => {
    await testReduce(
        'LogSum', {axes: [0], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
          ],
        },
        {
          shape: [2, 2],
          values: [
            2.48490665, 2.708050201,
            2.890371758, 3.044522438,
          ],
        });
  });

  it('reduceLogSum axes1 do not keep dims', async () => {
    await testReduce(
        'LogSum', {axes: [1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
          ],
        },
        {
          shape: [3, 2],
          values: [
            0.693147181, 1.386294361,
            2.302585093, 2.48490665,
            2.890371758, 2.995732274,
          ],
        });
  });

  it('reduceLogSum axes2 do not keep dims', async () => {
    await testReduce(
        'LogSum', {axes: [2], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        {
          shape: [3, 2],
          values: [
            0, 1.609437912,
            2.197224577, 2.564949357,
            2.833213344, 3.044522438,
          ],
        });
  });

  it('reduceLogSum axes0 keep dims', async () => {
    await testReduce(
        'LogSum', {axes: [0], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
          ],
        },
        {
          shape: [1, 2, 2],
          values: [
            2.48490665, 2.708050201,
            2.890371758, 3.044522438,
          ],
        });
  });

  it('reduceLogSum axes1 keep dims', async () => {
    await testReduce(
        'LogSum', {axes: [1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
          ],
        },
        {
          shape: [3, 1, 2],
          values: [
            0.693147181, 1.386294361,
            2.302585093, 2.48490665,
            2.890371758, 2.995732274,
          ],
        });
  });

  it('reduceLogSum axes2 keep dims', async () => {
    await testReduce(
        'LogSum', {axes: [2], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
          ],
        },
        {
          shape: [3, 2, 1],
          values: [
            0, 1.609437912,
            2.197224577, 2.564949357,
            2.833213344, 3.044522438,
          ],
        });
  });

  it('reduceLogSumExp default', async () => {
    await testReduce(
        'LogSumExp', {}, {
          shape: [3, 2, 2],
          values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        {shape: [], values: [11.458669]});
  });

  it('reduceLogSumExp default axes keep dims', async () => {
    await testReduce(
        'LogSumExp', {keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        {shape: [1, 1, 1], values: [11.458669]});
  });

  it('reduceLogSumExp axes0 do not keep dims', async () => {
    await testReduce(
        'LogSumExp', {axes: [0], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
          ],
        },
        {
          shape: [2, 2],
          values: [
            8.0184793, 9.0184793,
            10.0184793, 11.0184793,
          ],
        });
  });

  it('reduceLogSumExp axes1 do not keep dims', async () => {
    await testReduce(
        'LogSumExp', {axes: [1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
          ],
        },
        {
          shape: [3, 2],
          values: [
            2.12692801,
            3.12692801,
            6.12692801,
            7.12692801,
            10.12692801,
            11.12692801,
          ],
        });
  });

  it('reduceLogSumExp axes2 do not keep dims', async () => {
    await testReduce(
        'LogSumExp', {axes: [2], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
          ],
        },
        {
          shape: [3, 2],
          values: [
            1.31326169,
            3.31326169,
            5.31326169,
            7.31326169,
            9.31326169,
            11.31326169,
          ],
        });
  });

  it('reduceLogSumExp axes0 keep dims', async () => {
    await testReduce(
        'LogSumExp', {axes: [0], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
          ],
        },
        {
          shape: [1, 2, 2],
          values: [
            8.0184793, 9.0184793,
            10.0184793, 11.0184793,
          ],
        });
  });

  it('reduceLogSumExp axes1 keep dims', async () => {
    await testReduce(
        'LogSumExp', {axes: [1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        {
          shape: [3, 1, 2],
          values: [
            2.12692801,
            3.12692801,
            6.12692801,
            7.12692801,
            10.12692801,
            11.12692801,
          ],
        });
  });

  it('reduceLogSumExp axes2 keep dims', async () => {
    await testReduce(
        'LogSumExp', {axes: [2], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        {
          shape: [3, 2, 1],
          values: [
            1.31326169,
            3.31326169,
            5.31326169,
            7.31326169,
            9.31326169,
            11.31326169,
          ],
        });
  });

  it('reduceMax default', async () => {
    await testReduce(
        'Max', {}, {
          shape: [3, 2, 2],
          values: [1, 100, 200, 2, 300, 3, 4, 400, 500, 5, 600, 6],
        },
        {shape: [], values: [600]});
  });

  it('reduceMax default axes keep dims', async () => {
    await testReduce(
        'Max', {keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [1, 100, 200, 2, 300, 3, 4, 400, 500, 5, 600, 6],
        },
        {shape: [1, 1, 1], values: [600]});
  });

  it('reduceMax axes0 do not keep dims', async () => {
    await testReduce(
        'Max', {axes: [0], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [1, 100, 200, 2, 300, 3, 4, 400, 500, 5, 600, 6],
        },
        {shape: [2, 2], values: [500, 100, 600, 400]});
  });

  it('reduceMax axes1 do not keep dims', async () => {
    await testReduce(
        'Max', {axes: [1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [1, 100, 200, 2, 300, 3, 4, 400, 500, 5, 600, 6],
        },
        {shape: [3, 2], values: [200, 100, 300, 400, 600, 6]});
  });

  it('reduceMax axes2 do not keep dims', async () => {
    await testReduce(
        'Max', {axes: [2], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [1, 100, 200, 2, 300, 3, 4, 400, 500, 5, 600, 6],
        },
        {shape: [3, 2], values: [100, 200, 300, 400, 500, 600]});
  });

  it('reduceMax axes0 keep dims', async () => {
    await testReduce(
        'Max', {axes: [0], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [1, 100, 200, 2, 300, 3, 4, 400, 500, 5, 600, 6],
        },
        {shape: [1, 2, 2], values: [500, 100, 600, 400]});
  });

  it('reduceMax axes1 keep dims', async () => {
    await testReduce(
        'Max', {axes: [1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [1, 100, 200, 2, 300, 3, 4, 400, 500, 5, 600, 6],
        },
        {shape: [3, 1, 2], values: [200, 100, 300, 400, 600, 6]});
  });

  it('reduceMax axes2 keep dims', async () => {
    await testReduce(
        'Max', {axes: [2], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [1, 100, 200, 2, 300, 3, 4, 400, 500, 5, 600, 6],
        },
        {shape: [3, 2, 1], values: [100, 200, 300, 400, 500, 600]});
  });

  it('reduceMean default', async () => {
    await testReduce(
        'Mean', {}, {
          shape: [3, 2, 2],
          values: [5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2],
        },
        {shape: [], values: [18.25]});
  });

  it('reduceMean default axes keep dims', async () => {
    await testReduce(
        'Mean', {keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2],
        },
        {shape: [1, 1, 1], values: [18.25]});
  });

  it('reduceMean axes0 do not keep dims', async () => {
    await testReduce(
        'Mean', {axes: [0], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2],
        },
        {shape: [2, 2], values: [30, 1, 40, 2]});
  });

  it('reduceMean axes1 do not keep dims', async () => {
    await testReduce(
        'Mean', {axes: [1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2],
        },
        {shape: [3, 2], values: [12.5, 1.5, 35, 1.5, 57.5, 1.5]});
  });

  it('reduceMean axes2 do not keep dims', async () => {
    await testReduce(
        'Mean', {axes: [2], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2],
        },
        {shape: [3, 2], values: [3, 11, 15.5, 21, 28, 31]});
  });

  it('reduceMean axes0 keep dims', async () => {
    await testReduce(
        'Mean', {axes: [0], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2],
        },
        {shape: [1, 2, 2], values: [30, 1, 40, 2]});
  });

  it('reduceMean axes1 keep dims', async () => {
    await testReduce(
        'Mean', {axes: [1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2],
        },
        {shape: [3, 1, 2], values: [12.5, 1.5, 35, 1.5, 57.5, 1.5]});
  });

  it('reduceMean axes2 keep dims', async () => {
    await testReduce(
        'Mean', {axes: [2], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [5, 1, 20, 2, 30, 1, 40, 2, 55, 1, 60, 2],
        },
        {shape: [3, 2, 1], values: [3, 11, 15.5, 21, 28, 31]});
  });

  it('reduceMin default', async () => {
    await testReduce(
        'Min', {}, {
          shape: [3, 2, 2],
          values: [1, 100, 200, 2, 300, 3, 4, 400, 500, 5, 600, 6],
        },
        {shape: [], values: [1]});
  });

  it('reduceMin default axes keep dims', async () => {
    await testReduce(
        'Min', {keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [1, 100, 200, 2, 300, 3, 4, 400, 500, 5, 600, 6],
        },
        {shape: [1, 1, 1], values: [1]});
  });

  it('reduceMin axes0 do not keep dims', async () => {
    await testReduce(
        'Min', {axes: [0], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [1, 100, 200, 2, 300, 3, 4, 400, 500, 5, 600, 6],
        },
        {shape: [2, 2], values: [1, 3, 4, 2]});
  });

  it('reduceMin axes1 do not keep dims', async () => {
    await testReduce(
        'Min', {axes: [1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [1, 100, 200, 2, 300, 3, 4, 400, 500, 5, 600, 6],
        },
        {shape: [3, 2], values: [1, 2, 4, 3, 500, 5]});
  });

  it('reduceMin axes2 do not keep dims', async () => {
    await testReduce(
        'Min', {axes: [2], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [1, 100, 200, 2, 300, 3, 4, 400, 500, 5, 600, 6],
        },
        {shape: [3, 2], values: [1, 2, 3, 4, 5, 6]});
  });

  it('reduceMin axes0 keep dims', async () => {
    await testReduce(
        'Min', {axes: [0], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [1, 100, 200, 2, 300, 3, 4, 400, 500, 5, 600, 6],
        },
        {shape: [1, 2, 2], values: [1, 3, 4, 2]});
  });

  it('reduceMin axes1 keep dims', async () => {
    await testReduce(
        'Min', {axes: [1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [1, 100, 200, 2, 300, 3, 4, 400, 500, 5, 600, 6],
        },
        {shape: [3, 1, 2], values: [1, 2, 4, 3, 500, 5]});
  });

  it('reduceMin axes2 keep dims', async () => {
    await testReduce(
        'Min', {axes: [2], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [1, 100, 200, 2, 300, 3, 4, 400, 500, 5, 600, 6],
        },
        {shape: [3, 2, 1], values: [1, 2, 3, 4, 5, 6]});
  });

  it('reduceProduct default', async () => {
    await testReduce(
        'Product', {}, {
          shape: [3, 2, 2],
          values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        {shape: [], values: [0]});
  });

  it('reduceProduct default axes keep dims', async () => {
    await testReduce(
        'Product', {keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        {shape: [1, 1, 1], values: [0]});
  });

  it('reduceProduct axes0 do not keep dims', async () => {
    await testReduce(
        'Product', {axes: [0], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        {
          shape: [2, 2],
          values: [0, 45, 120, 231],
        });
  });

  it('reduceProduct axes1 do not keep dims', async () => {
    await testReduce(
        'Product', {axes: [1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        {
          shape: [3, 2],
          values: [0, 3, 24, 35, 80, 99],
        });
  });

  it('reduceProduct axes2 do not keep dims', async () => {
    await testReduce(
        'Product', {axes: [2], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        {
          shape: [3, 2],
          values: [0, 6, 20, 42, 72, 110],
        });
  });

  it('reduceProduct axes0 keep dims', async () => {
    await testReduce(
        'Product', {axes: [0], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        {
          shape: [1, 2, 2],
          values: [0, 45, 120, 231],
        });
  });

  it('reduceProduct axes1 keep dims', async () => {
    await testReduce(
        'Product', {axes: [1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        {
          shape: [3, 1, 2],
          values: [0, 3, 24, 35, 80, 99],
        });
  });

  it('reduceProduct axes2 keep dims', async () => {
    await testReduce(
        'Product', {axes: [2], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        {
          shape: [3, 2, 1],
          values: [0, 6, 20, 42, 72, 110],
        });
  });

  it('reduceSum default', async () => {
    await testReduce(
        'Sum', {}, {
          shape: [3, 2, 2],
          values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        {shape: [], values: [66]});
  });

  it('reduceSum default axes keep dims', async () => {
    await testReduce(
        'Sum', {keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        {shape: [1, 1, 1], values: [66]});
  });

  it('reduceSum axes0 do not keep dims', async () => {
    await testReduce(
        'Sum', {axes: [0], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        {
          shape: [2, 2],
          values: [12, 15, 18, 21],
        });
  });

  it('reduceSum axes1 do not keep dims', async () => {
    await testReduce(
        'Sum', {axes: [1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        {
          shape: [3, 2],
          values: [2, 4, 10, 12, 18, 20],
        });
  });

  it('reduceSum axes2 do not keep dims', async () => {
    await testReduce(
        'Sum', {axes: [2], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        {
          shape: [3, 2],
          values: [1, 5, 9, 13, 17, 21],
        });
  });

  it('reduceSum axes0 keep dims', async () => {
    await testReduce(
        'Sum', {axes: [0], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        {
          shape: [1, 2, 2],
          values: [12, 15, 18, 21],
        });
  });

  it('reduceSum axes1 keep dims', async () => {
    await testReduce(
        'Sum', {axes: [1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        {
          shape: [3, 1, 2],
          values: [2, 4, 10, 12, 18, 20],
        });
  });

  it('reduceSum axes2 keep dims', async () => {
    await testReduce(
        'Sum', {axes: [2], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        {
          shape: [3, 2, 1],
          values: [1, 5, 9, 13, 17, 21],
        });
  });

  it('reduceSumSquare default', async () => {
    await testReduce(
        'SumSquare', {}, {
          shape: [3, 2, 2],
          values: [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
          ],
        },
        {shape: [], values: [506]});
  });

  it('reduceSumSquare default axes keep dims', async () => {
    await testReduce(
        'SumSquare', {keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
          ],
        },
        {shape: [1, 1, 1], values: [506]});
  });

  it('reduceSumSquare axes0 do not keep dims', async () => {
    await testReduce(
        'SumSquare', {axes: [0], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
          ],
        },
        {
          shape: [2, 2],
          values: [80, 107, 140, 179],
        });
  });

  it('reduceSumSquare axes1 do not keep dims', async () => {
    await testReduce(
        'SumSquare', {axes: [1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
          ],
        },
        {
          shape: [3, 2],
          values: [4, 10, 52, 74, 164, 202],
        });
  });

  it('reduceSumSquare axes2 do not keep dims', async () => {
    await testReduce(
        'SumSquare', {axes: [2], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
          ],
        },
        {
          shape: [3, 2],
          values: [1, 13, 41, 85, 145, 221],
        });
  });

  it('reduceSumSquare axes0 keep dims', async () => {
    await testReduce(
        'SumSquare', {axes: [0], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
          ],
        },
        {
          shape: [1, 2, 2],
          values: [80, 107, 140, 179],
        });
  });

  it('reduceSumSquare axes1 keep dims', async () => {
    await testReduce(
        'SumSquare', {axes: [1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
          ],
        },
        {
          shape: [3, 1, 2],
          values: [4, 10, 52, 74, 164, 202],
        });
  });

  it('reduceSumSquare axes2 keep dims', async () => {
    await testReduce(
        'SumSquare', {axes: [2], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0, 1, 2, 3,
            4, 5, 6, 7,
            8, 9, 10, 11,
          ],
        },
        {
          shape: [3, 2, 1],
          values: [1, 13, 41, 85, 145, 221],
        });
  });

  it('reduceL1 default', async () => {
    await testReduce(
        'L1', {}, {
          shape: [3, 2, 2],
          values: [
            0.9762701,  4.303787,
            2.0552676,  0.89766365,
            -1.526904,  2.9178822,
            -1.2482557, 7.83546,
            9.273255,   -2.3311696,
            5.834501,   0.5778984,
          ],
        },
        {shape: [], values: [39.778313]});
  });

  it('reduceL1 default axes keep dims', async () => {
    await testReduce(
        'L1', {keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0.9762701,  4.303787,
            2.0552676,  0.89766365,
            -1.526904,  2.9178822,
            -1.2482557, 7.83546,
            9.273255,   -2.3311696,
            5.834501,   0.5778984,
          ],
        },
        {shape: [1, 1, 1], values: [39.778313]});
  });

  it('reduceL1 axes0 do not keep dims', async () => {
    await testReduce(
        'L1', {axes: [0], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0.9762701,  4.303787,
            2.0552676,  0.89766365,
            -1.526904,  2.9178822,
            -1.2482557, 7.83546,
            9.273255,   -2.3311696,
            5.834501,   0.5778984,
          ],
        },
        {
          shape: [2, 2],
          values: [
            11.776429, 9.552839,
            9.138024,  9.311022,
          ],
        });
  });

  it('reduceL1 axes1 do not keep dims', async () => {
    await testReduce(
        'L1', {axes: [1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0.9762701,  4.303787,
            2.0552676,  0.89766365,
            -1.526904,  2.9178822,
            -1.2482557, 7.83546,
            9.273255,   -2.3311696,
            5.834501,   0.5778984,
          ],
        },
        {
          shape: [3, 2],
          values: [
            3.0315375, 5.201451,
            2.7751598, 10.753343,
            15.107756, 2.909068,
          ],
        });
  });

  it('reduceL1 axes2 do not keep dims', async () => {
    await testReduce(
        'L1', {axes: [2], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0.9762701,  4.303787,
            2.0552676,  0.89766365,
            -1.526904,  2.9178822,
            -1.2482557, 7.83546,
            9.273255,   -2.3311696,
            5.834501,   0.5778984,
          ],
        },
        {
          shape: [3, 2],
          values: [
            5.2800574, 2.9529312,
            4.444786,  9.083715,
            11.604425, 6.4123993,
          ],
        });
  });

  it('reduceL1 axes0 keep dims', async () => {
    await testReduce(
        'L1', {axes: [0], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0.9762701,  4.303787,
            2.0552676,  0.89766365,
            -1.526904,  2.9178822,
            -1.2482557, 7.83546,
            9.273255,   -2.3311696,
            5.834501,   0.5778984,
          ],
        },
        {
          shape: [1, 2, 2],
          values: [
            11.776429, 9.552839,
            9.138024,  9.311022,
          ],
        });
  });

  it('reduceL1 axes1 keep dims', async () => {
    await testReduce(
        'L1', {axes: [1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0.9762701,  4.303787,
            2.0552676,  0.89766365,
            -1.526904,  2.9178822,
            -1.2482557, 7.83546,
            9.273255,   -2.3311696,
            5.834501,   0.5778984,
          ],
        },
        {
          shape: [3, 1, 2],
          values: [
            3.0315375, 5.201451,
            2.7751598, 10.753343,
            15.107756, 2.909068,
          ],
        });
  });

  it('reduceL1 axes2 keep dims', async () => {
    await testReduce(
        'L1', {axes: [2], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0.9762701,  4.303787,
            2.0552676,  0.89766365,
            -1.526904,  2.9178822,
            -1.2482557, 7.83546,
            9.273255,   -2.3311696,
            5.834501,   0.5778984,
          ],
        },
        {
          shape: [3, 2, 1],
          values: [
            5.2800574, 2.9529312,
            4.444786,  9.083715,
            11.604425, 6.4123993,
          ],
        });
  });

  it('reduceL2 default', async () => {
    await testReduce(
        'L2', {}, {
          shape: [3, 2, 2],
          values: [
            0.9762701,  4.303787,
            2.0552676,  0.89766365,
            -1.526904,  2.9178822,
            -1.2482557, 7.83546,
            9.273255,   -2.3311696,
            5.834501,   0.5778984,
          ],
        },
        {shape: [], values: [14.970192]});
  });

  it('reduceL2 default axes keep dims', async () => {
    await testReduce(
        'L2', {keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0.9762701,  4.303787,
            2.0552676,  0.89766365,
            -1.526904,  2.9178822,
            -1.2482557, 7.83546,
            9.273255,   -2.3311696,
            5.834501,   0.5778984,
          ],
        },
        {shape: [1, 1, 1], values: [14.970192]});
  });

  it('reduceL2 axes0 do not keep dims', async () => {
    await testReduce(
        'L2', {axes: [0], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0.9762701,  4.303787,
            2.0552676,  0.89766365,
            -1.526904,  2.9178822,
            -1.2482557, 7.83546,
            9.273255,   -2.3311696,
            5.834501,   0.5778984,
          ],
        },
        {
          shape: [2, 2],
          values: [
            9.448693, 5.698331,
            6.3106,   7.907857,
          ],
        });
  });

  it('reduceL2 axes1 do not keep dims', async () => {
    await testReduce(
        'L2', {axes: [1], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0.9762701,  4.303787,
            2.0552676,  0.89766365,
            -1.526904,  2.9178822,
            -1.2482557, 7.83546,
            9.273255,   -2.3311696,
            5.834501,   0.5778984,
          ],
        },
        {
          shape: [3, 2],
          values: [
            2.2753522,  4.3964057,
            1.9722013,  8.361129,
            10.956034,  2.4017324,
          ],
        });
  });

  it('reduceL2 axes2 do not keep dims', async () => {
    await testReduce(
        'L2', {axes: [2], keepDimensions: false}, {
          shape: [3, 2, 2],
          values: [
            0.9762701,  4.303787,
            2.0552676,  0.89766365,
            -1.526904,  2.9178822,
            -1.2482557, 7.83546,
            9.273255,   -2.3311696,
            5.834501,   0.5778984,
          ],
        },
        {
          shape: [3, 2],
          values: [
            4.413127,  2.2427495,
            3.2932465, 7.934266,
            9.561779,  5.863051,
          ],
        });
  });

  it('reduceL2 axes0 keep dims', async () => {
    await testReduce(
        'L2', {axes: [0], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0.9762701,  4.303787,
            2.0552676,  0.89766365,
            -1.526904,  2.9178822,
            -1.2482557, 7.83546,
            9.273255,   -2.3311696,
            5.834501,   0.5778984,
          ],
        },
        {
          shape: [1, 2, 2],
          values: [
            9.448693, 5.698331,
            6.3106,   7.907857,
          ],
        });
  });

  it('reduceL2 axes1 keep dims', async () => {
    await testReduce(
        'L2', {axes: [1], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0.9762701,  4.303787,
            2.0552676,  0.89766365,
            -1.526904,  2.9178822,
            -1.2482557, 7.83546,
            9.273255,   -2.3311696,
            5.834501,   0.5778984,
          ],
        },
        {
          shape: [3, 1, 2],
          values: [
            2.2753522,  4.3964057,
            1.9722013,  8.361129,
            10.956034,  2.4017324,
          ],
        });
  });

  it('reduceL2 axes2 keep dims', async () => {
    await testReduce(
        'L2', {axes: [2], keepDimensions: true}, {
          shape: [3, 2, 2],
          values: [
            0.9762701,  4.303787,
            2.0552676,  0.89766365,
            -1.526904,  2.9178822,
            -1.2482557, 7.83546,
            9.273255,   -2.3311696,
            5.834501,   0.5778984,
          ],
        },
        {
          shape: [3, 2, 1],
          values: [
            4.413127,  2.2427495,
            3.2932465, 7.934266,
            9.561779,  5.863051,
          ],
        });
  });
});
