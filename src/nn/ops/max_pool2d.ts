import {Pool} from './pool';

export class MaxPool2d extends Pool {
  getPoolingType(): 'avg'|'max' {
    return 'max';
  }
}