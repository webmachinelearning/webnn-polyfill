global.navigator = {};
require('../dist/webnn-polyfill.js');
require('@babel/register')({
  only: [/test/],
});
require('@babel/polyfill');
global.chai = require('chai');
