module.exports = {
  root: true,
  ignorePatterns: ['**/*.ts', 'node_modules/', 'dist/', 'docs/', 'webpack.config.js', 'test/lib/', '.eslintrc.js', 'test/cts/from_nnapi/tests/cts.js'],
  env: { 'es6': true, 'browser': true, 'node': true, 'mocha': true },
  parserOptions: { ecmaVersion: 2020, sourceType: 'module'},
  globals: {
    'chai': 'readonly',
    'ML': 'readonly',
    'MLContext': 'readonly',
    'MLGraphBuilder': 'readonly',
    'MLGraph': 'readonly',
    'MLOperand': 'readonly',
    '_tfengine': 'readonly',
    'BigInt': 'readonly',
    'BigInt64Array': 'readonly',
    'BigUint64Array': 'readonly',
    'fs': 'readonly',
    'numpy': 'readonly'
  },
  rules: {
    'semi': 'error',
    'no-multi-spaces': ['error', { 'exceptions': { 'ArrayExpression': true } }],
    'indent': 2,
    'require-jsdoc': 'off',
  },
  extends: [
    'eslint:recommended',
    'google',
  ],
}
