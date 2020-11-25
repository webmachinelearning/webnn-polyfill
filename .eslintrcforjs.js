module.exports = {
  root: true,
  ignorePatterns: ['**/*.ts', 'node_modules/', 'dist/', 'docs/', 'webpack.config.js', 'test/numpy.js'],
  env: { 'es6': true, 'browser': true, 'node': true, 'mocha': true},
  parserOptions: { ecmaVersion: 2017, sourceType: 'module'},
  globals: {
    'chai': 'readonly',
    'ML': 'readonly',
    'NeuralNetworkContext': 'readonly',
    'ModelBuilder': 'readonly',
    'Model': 'readonly',
    'Compilation': 'readonly',
    'Operand': 'readonly',
    '_tfengine': 'readonly',
    'BigInt64Array': 'readonly',
    'BigUint64Array': 'readonly',
    'fs': 'readonly',
    'numpy': 'readonly'
  },
  rules: {
    'semi': 'error',
    'no-multi-spaces': ['error', { 'exceptions': { 'ArrayExpression': true } }],
    'indent': 'off',
    'require-jsdoc': 'off',
  },
  extends: [
    'eslint:recommended',
    'google',
  ],
}
