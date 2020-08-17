const assert = chai.assert;

function almostEqual(a, b, episilon=1e-6, rtol=5.0*1.1920928955078125e-7) {
  let delta = Math.abs(a - b);
  if (delta <= episilon + rtol * Math.abs(b)) {
    return true;
  } else {
    console.warn(`a(${a}) b(${b}) delta(${delta})`);
    return false;
  }
}

function checkOutput(output, expected) {
  assert.isTrue(output.length === expected.length);
  for (let i = 0; i < output.length; ++i) {
    assert.isTrue(almostEqual(output[i], expected[i]));
  }
}
