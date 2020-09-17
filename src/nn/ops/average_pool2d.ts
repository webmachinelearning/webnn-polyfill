import {Pool} from './pool';

export class AveragePool2d extends Pool {
  getPoolingType(): 'avg'|'max' {
    return 'avg';
  }
}