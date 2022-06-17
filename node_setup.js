global.navigator = {};
require('./dist/webnn-polyfill.js');
global.chai = require('chai');
global.fs = require('fs');

exports.mochaGlobalSetup = async function() {
  // Set 'cpu' as default backend for `npm test`
  const context = await navigator.ml.createContext();
  const tf = context.tf;
  await tf.setBackend('cpu');
  await tf.ready();
};
