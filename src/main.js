import {ML} from './ml';
import {Compilation} from './nn/compilation';
import {Model} from './nn/model';
import {ModelBuilder} from './nn/model_builder';
import {NeuralNetworkContext} from './nn/neural_network_context';
import {Operand} from './nn/operand';

if (navigator.ml == null) {
  navigator.ml = new ML();
}

if (global.ML == null) {
  global.ML = ML;
}

if (global.NeuralNetworkContext == null) {
  global.NeuralNetworkContext = NeuralNetworkContext;
}

if (global.ModelBuilder == null) {
  global.ModelBuilder = ModelBuilder;
}

if (global.Model == null) {
  global.Model = Model;
}

if (global.Compilation == null) {
  global.Compilation = Compilation;
}

if (global.Operand == null) {
  global.Operand = Operand;
}
