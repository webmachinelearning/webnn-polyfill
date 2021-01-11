'use strict';
const assert = chai.assert;

class AccuracyCriterion {
  constructor(atol, rtol) {
    this.atol = atol;
    this.rtol = rtol;
  }
}

export const opFp32AccuracyCriteria =
  new AccuracyCriterion(1e-6, 5.0 * 1.1920928955078125e-7);

// The following 2 constants were used for converted tests from NNAPI CTS
export const ctsFp32RestrictAccuracyCriteria =
  new AccuracyCriterion(1e-5, 5.0 * 1.1920928955078125e-7);
export const ctsFp32RelaxedAccuracyCriteria =
  new AccuracyCriterion(5.0 * 0.0009765625, 5.0 * 0.0009765625);

export function almostEqual(a, b, criteria) {
  const delta = Math.abs(a - b);
  if (delta <= criteria.atol + criteria.rtol * Math.abs(b)) {
    return true;
  } else {
    console.warn(`a(${a}) b(${b}) delta(${delta})`);
    return false;
  }
}

export function checkValue(
    output, expected, criteria = opFp32AccuracyCriteria) {
  assert.isTrue(output.length === expected.length);
  for (let i = 0; i < output.length; ++i) {
    assert.isTrue(almostEqual(output[i], expected[i], criteria));
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
