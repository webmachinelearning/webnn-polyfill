import {ML} from './ML';

if (navigator.ml == null) {
  navigator.ml = new ML();
}
