'use strict';
const assert = chai.assert;

// The following 4 constants were used for converted tests from NNAPI CTS
export const atol = 1e-5;
export const rtol = 5.0 * 1.1920928955078125e-7;
export const atolRelaxed = 5.0 * 0.0009765625;
export const rtolRelaxed = 5.0 * 0.0009765625;

export function almostEqual(a, b, episilon, rtol) {
  const delta = Math.abs(a - b);
  if (delta <= episilon + rtol * Math.abs(b)) {
    return true;
  } else {
    console.warn(`a(${a}) b(${b}) delta(${delta})`);
    return false;
  }
}

export function checkValue(
    output, expected, episilon = 1e-6, rtol = 5.0 * 1.1920928955078125e-7) {
  assert.isTrue(output.length === expected.length);
  for (let i = 0; i < output.length; ++i) {
    assert.isTrue(almostEqual(output[i], expected[i], episilon, rtol));
  }
}

export function sizeOfShape(array) {
  return array.reduce(
      (accumulator, currentValue) => accumulator * currentValue);
}

export function checkShape(shape, expected) {
  assert.isTrue(shape.length === expected.length);
  for (let i = 0; i < shape.length; ++i) {
    assert.isTrue(shape[i] === expected[i]);
  }
}
