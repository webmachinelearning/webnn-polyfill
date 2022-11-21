global.navigator = {};
require('./dist/webnn-polyfill.js');
global.chai = require('chai');
const chaiAsPromised = require('chai-as-promised');
global.chai.use(chaiAsPromised);
global.fs = require('fs');

exports.mochaGlobalSetup = async () => {
  // Set 'cpu' as default backend for `npm test`
  const context = await navigator.ml.createContext();
  const tf = context.tf;
  await tf.setBackend('cpu');
  await tf.ready();
};
