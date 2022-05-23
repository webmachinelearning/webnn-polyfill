'use strict';
import * as utils from '../utils.js';

describe('test batchNormalization', function() {
  const context = navigator.ml.createContext();

  function testBatchNorm(
      input, mean, variance, expected, scale = undefined, bias = undefined,
      options = {}, activation = undefined) {
    const builder = new MLGraphBuilder(context);
    const x =
        builder.input('input', {type: 'float32', dimensions: input.shape});
    const m =
        builder.constant({type: 'float32', dimensions: mean.shape}, mean.data);
    const v = builder.constant(
        {type: 'float32', dimensions: variance.shape}, variance.data);
    if (scale !== undefined) {
      options.scale = builder.constant(
          {type: 'float32', dimensions: scale.shape}, scale.data);
    }
    if (bias !== undefined) {
      options.bias = builder.constant(
          {type: 'float32', dimensions: bias.shape}, bias.data);
    }
    if (activation !== undefined) {
      options.activation = utils.createActivation(builder, activation);
    }
    const output = builder.batchNormalization(x, m, v, options);
    const graph = builder.build({output});
    const inputs = {'input': input.data};
    const outputs = {
      'output': new Float32Array(utils.sizeOfShape(input.shape)),
    };
    graph.compute(inputs, outputs);
    utils.checkValue(outputs.output, expected);
  }

  it('batchNormalization nchw', function() {
    const input = {
      shape: [1, 2, 1, 3],
      data: new Float32Array([-1, 0, 1, 2, 3, 4]),
    };
    const mean = {
      shape: [2],
      data: new Float32Array([0, 3]),
    };
    const variance = {
      shape: [2],
      data: new Float32Array([1.0, 1.5]),
    };
    const scale = {
      shape: [2],
      data: new Float32Array([1.0, 1.5]),
    };
    const bias = {
      shape: [2],
      data: new Float32Array([0, 1]),
    };
    let expected = [-0.999995, 0., 0.999995, -0.22474074, 1., 2.2247407];
    testBatchNorm(input, mean, variance, expected, scale, bias);

    expected = [0., 0., 0.999995, 0., 1., 2.2247407];
    testBatchNorm(input, mean, variance, expected, scale, bias, {}, 'relu');

    let expectedScale = [-0.999995, 0., 0.999995, -1.22474, 0., 1.22474];
    testBatchNorm(input, mean, variance, expectedScale, scale);

    expectedScale = [0., 0., 0.999995, 0., 0., 1.22474];
    testBatchNorm(
        input, mean, variance, expectedScale, scale, undefined, {}, 'relu');

    let expectedBias = [-0.999995, 0., 0.999995, 0.183506, 1., 1.816494];
    testBatchNorm(input, mean, variance, expectedBias, undefined, bias);

    expectedBias = [0., 0., 0.999995, 0.183506, 1., 1.816494];
    testBatchNorm(
        input, mean, variance, expectedBias, undefined, bias, {}, 'relu');
  });

  it('batchNormalization 3D input axis=0', function() {
    const input = {
      shape: [3, 1, 2],
      data: new Float32Array([-1, 0, 1, 2, 3, 4]),
    };
    const mean = {
      shape: [3],
      data: new Float32Array([0, 3, 6]),
    };
    const variance = {
      shape: [3],
      data: new Float32Array([1.0, 1.5, 2.0]),
    };
    const scale = {
      shape: [3],
      data: new Float32Array([1.0, 1.5, 2.0]),
    };
    const bias = {
      shape: [3],
      data: new Float32Array([0, 1, 2]),
    };
    const expected = [
      -0.9995003746877732,
      0,
      -1.4486736542238683,
      -0.22433682711193415,
      -2.241580424529414,
      -0.8277202830196093,
    ];
    testBatchNorm(input, mean, variance, expected, scale, bias,
        {epsilon: 1e-3, axis: 0});
  });

  it('batchNormalization 3D input axis=2', function() {
    const input = {
      shape: [2, 1, 3],
      data: new Float32Array([-1, 0, 1, 2, 3, 4]),
    };
    const mean = {
      shape: [3],
      data: new Float32Array([0, 3, 6]),
    };
    const variance = {
      shape: [3],
      data: new Float32Array([1.0, 1.5, 2.0]),
    };
    const scale = {
      shape: [3],
      data: new Float32Array([1.0, 1.5, 2.0]),
    };
    const bias = {
      shape: [3],
      data: new Float32Array([0, 1, 2]),
    };
    const expected = [
      -0.9995003746877732,
      -2.6730104813358024,
      -5.069300707549023,
      1.9990007493755464,
      1,
      -0.8277202830196093,
    ];
    testBatchNorm(input, mean, variance, expected, scale, bias,
        {epsilon: 1e-3, axis: 2});
  });

  it('batchNormalization nhwc', function() {
    const input = {
      shape: [1, 1, 3, 2],
      data: new Float32Array([-1, 2, 0, 3, 1, 4]),
    };
    const mean = {
      shape: [2],
      data: new Float32Array([0, 3]),
    };
    const variance = {
      shape: [2],
      data: new Float32Array([1.0, 1.5]),
    };
    const scale = {
      shape: [2],
      data: new Float32Array([1.0, 1.5]),
    };
    const bias = {
      shape: [2],
      data: new Float32Array([0, 1]),
    };
    let expected = [-0.999995, -0.22474074, 0., 1., 0.999995, 2.2247407];
    testBatchNorm(input, mean, variance, expected, scale, bias, {axis: 3});

    expected = [0., 0., 0., 1., 0.999995, 2.2247407];
    testBatchNorm(
        input, mean, variance, expected, scale, bias, {axis: 3}, 'relu');

    let expectedScale = [-0.999995, -1.22474, 0., 0., 0.999995, 1.22474];
    testBatchNorm(
        input, mean, variance, expectedScale, scale, undefined, {axis: 3});

    expectedScale = [0., 0., 0., 0., 0.999995, 1.22474];
    testBatchNorm(
        input, mean, variance, expectedScale, scale, undefined, {axis: 3},
        'relu');

    let expectedBias = [-0.999995, 0.183506, 0., 1., 0.999995, 1.816494];
    testBatchNorm(
        input, mean, variance, expectedBias, undefined, bias, {axis: 3});

    expectedBias = [0., 0.183506, 0., 1., 0.999995, 1.816494];
    testBatchNorm(
        input, mean, variance, expectedBias, undefined, bias, {axis: 3},
        'relu');
  });

  it('batchNormalization without options', function() {
    const input = {
      shape: [1, 2, 1, 3],
      data: new Float32Array([-1, 0, 1, 2, 3, 4]),
    };
    const mean = {
      shape: [2],
      data: new Float32Array([0, 3]),
    };
    const variance = {
      shape: [2],
      data: new Float32Array([1.0, 1.5]),
    };

    const expected = [-0.999995, 0., 0.999995, -0.816494, 0., 0.816494];
    testBatchNorm(input, mean, variance, expected);
  });

  it('batchNormalization with epsilon', function() {
    const input = {
      shape: [2, 3, 4, 5],
      data: new Float32Array([
        2.6973534,   -1.1874187,  -0.18637535, -1.7081367,  0.03293341,
        1.4802791,   -0.68332213, 1.618039,    -1.6412221,  -0.52998835,
        1.5229957,   -0.92798537, -0.35554567, 0.717948,    0.50108916,
        1.0521007,   -0.68065745, 1.3121722,   0.50907123,  1.5093223,
        -0.540522,   -0.80794656, -0.17974755, -1.8922086,  2.0955374,
        0.46592507,  -0.2936382,  -0.43420887, -0.11036888, -1.2171484,
        -1.9003569,  0.32063156,  0.38756344,  0.4720109,   -0.4177193,
        -0.7655141,  -1.2207903,  0.52860916,  0.22583283,  1.2220219,
        -0.0248001,  0.6148501,   1.0967597,   0.8798244,   -0.6854243,
        -0.8442876,  1.6188551,   -0.6460473,  0.76349306,  2.630077,
        -0.85050315, 0.37401453,  0.08842833,  -0.5043717,  -0.7495827,
        -0.98900026, 0.79681706,  -0.3573076,  0.8644746,   1.196009,
        0.35148722,  0.39926755,  -0.21630785, 1.731195,    1.8644739,
        -0.60227305, -1.0833911,  -0.6197943,  -0.05721893, -0.23889631,
        -0.24901256, 1.3885167,   -0.67789817, -0.3381054,  0.33224156,
        0.79065573,  1.1667213,   -0.47722074, 0.4234017,   0.2317288,
        -0.18525974, -0.17303231, 0.41841915,  0.13230574,  0.1261528,
        1.253214,    1.9984859,   -1.7275336,  0.6593169,   -1.3704892,
        0.63530993,  -0.33128706, -1.2268444,  0.87340677,  1.4801403,
        0.09598545,  0.30467814,  -0.15848571, -0.16779709, 1.1372787,
        0.3292992,   -0.2240395,  0.88280654,  1.3370756,   0.2533313,
        0.84305125,  -1.6560661,  -0.09365056, -1.301057,   -0.1476929,
        -1.2850751,  -1.286735,   -1.9894414,  -0.5574838,  -0.392564,
        -0.92764777, -0.79910755, 0.9099533,   0.9825949,   -0.8327678,
      ]),
    };
    const mean = {
      shape: [3],
      data: new Float32Array([0.3432895, 1.0855169, 1.8725895]),
    };
    const variance = {
      shape: [3],
      data: new Float32Array([0.601868, 0.86580527, 0.38809904]),
    };
    const scale = {
      shape: [3],
      data: new Float32Array([0.17215693, -0.7909758, 0.12456307]),
    };
    const bias = {
      shape: [3],
      data: new Float32Array([0.5280557, -1.4475446, 0.1760742]),
    };
    const expected = [
      1.0461562e+00,  1.9116578e-01,  4.1148305e-01,  7.6562166e-02,
      4.5975018e-01,  7.7829313e-01,  3.0211121e-01,  8.0861235e-01,
      9.1289252e-02,  3.3585808e-01,  7.8769445e-01,  2.4826387e-01,
      3.7425077e-01,  6.1051345e-01,  5.6278551e-01,  6.8405628e-01,
      3.0269766e-01,  7.4129486e-01,  5.6454223e-01,  7.8468513e-01,
      -7.3216558e-02, 1.5281045e-01,  -3.7814319e-01, 1.0692286e+00,
      -2.3012137e+00, -9.2386562e-01, -2.8188276e-01, -1.6307259e-01,
      -4.3678200e-01, 4.9866784e-01,  1.0761156e+00,  -8.0106354e-01,
      -8.5763437e-01, -9.2900938e-01, -1.7700946e-01, 1.1694658e-01,
      5.0174618e-01,  -9.7684616e-01, -7.2093964e-01, -1.5629185e+00,
      -1.9851068e-01, -7.2230190e-02, 2.2908971e-02,  -1.9918650e-02,
      -3.2893193e-01, -3.6029488e-01, 1.2598167e-01,  -3.2115802e-01,
      -4.2884916e-02, 3.2561827e-01,  -3.6152196e-01, -1.1977625e-01,
      -1.7615700e-01, -2.9318827e-01, -3.4159809e-01, -3.8886422e-01,
      -3.6306053e-02, -2.6415467e-01, -2.2949010e-02, 4.2502895e-02,
      5.2985996e-01,  5.4037583e-01,  4.0489528e-01,  8.3351660e-01,
      8.6284959e-01,  3.1994909e-01,  2.1406096e-01,  3.1609288e-01,
      4.3990877e-01,  3.9992383e-01,  3.9769736e-01,  7.5809729e-01,
      3.0330497e-01,  3.7808913e-01,  5.2562422e-01,  6.2651551e-01,
      7.0928288e-01,  3.4747157e-01,  5.4568744e-01,  5.0350261e-01,
      -3.7348437e-01, -3.8381898e-01, -8.8371360e-01, -6.4189059e-01,
      -6.3669008e-01, -1.5892822e+00, -2.2191858e+00, 9.3004560e-01,
      -1.0873203e+00, 6.2827158e-01,  -1.0670297e+00, -2.5006199e-01,
      5.0686288e-01,  -1.2682691e+00, -1.7810802e+00, -6.1119264e-01,
      -7.8757972e-01, -3.9611375e-01, -3.8824368e-01, -1.4912935e+00,
      -1.2860399e-01, -2.3784474e-01, -1.9329906e-02, 7.0352428e-02,
      -1.4360166e-01, -2.7178437e-02, -5.2055711e-01, -2.1210325e-01,
      -4.5047081e-01, -2.2277233e-01, -4.4731563e-01, -4.4764340e-01,
      -5.8637255e-01, -3.0367374e-01, -2.7111509e-01, -3.7675196e-01,
      -3.5137540e-01, -1.3970569e-02, 3.7042797e-04,  -3.5802066e-01,
    ];
    testBatchNorm(
        input, mean, variance, expected, scale, bias, {epsilon: 1e-2});
  });
});
