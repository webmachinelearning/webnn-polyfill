module.exports = {
  root: true,
  ignorePatterns: ['**/*.ts', 'node_modules/', 'dist/', 'docs/', 'webpack.config.js'],
  env: { 'es6': true, 'browser': true, 'node': true, 'mocha': true},
  parserOptions: { ecmaVersion: 2017, sourceType: 'module'},
  globals: {
    'assert': 'readonly',
    'chai': 'readonly',
    'checkOutput': 'readonly'
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
