import {ML} from './ml';
import {MLContext} from './nn/context';
import {MLGraph} from './nn/graph';
import {MLGraphBuilder} from './nn/graph_builder';
import {MLOperand} from './nn/operand';

// for running in Node.js
if (typeof navigator == 'undefined') {
  global.navigator = {};
}

if (navigator.ml == null) {
  navigator.ml = new ML();
}

if (global.ML == null) {
  global.ML = ML;
}

if (global.MLContext == null) {
  global.MLContext = MLContext;
}

if (global.MLGraphBuilder == null) {
  global.MLGraphBuilder = MLGraphBuilder;
}

if (global.MLGraph == null) {
  global.MLGraph = MLGraph;
}

if (global.MLOperand == null) {
  global.MLOperand = MLOperand;
}
