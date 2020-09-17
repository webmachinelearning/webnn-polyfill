import {ML} from './ml';

if (navigator.ml == null) {
  navigator.ml = new ML();
}
