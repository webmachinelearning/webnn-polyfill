[![build and test](https://github.com/webmachinelearning/webnn-polyfill/workflows/build%20and%20test/badge.svg)](https://github.com/webmachinelearning/webnn-polyfill/actions)
[![deploy](https://github.com/webmachinelearning/webnn-polyfill/workflows/deploy/badge.svg)](https://github.com/webmachinelearning/webnn-polyfill/actions)

# WebNN Polyfill

A JavaScript implementation of the [Web Neural Network API](https://webmachinelearning.github.io/webnn/).

* [API docs](https://webmachinelearning.github.io/webnn-polyfill/docs/)
* [Unit tests](https://webmachinelearning.github.io/webnn-polyfill/test/)

## Backends

The implementation of this webnn-polyfill is based on [TensorFlow.js](https://github.com/tensorflow/tfjs) that supports the following 4 backends:

* [TensorFlow.js CPU Backend](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-cpu), pure-JS backend for Node.js and the browser.
* [TensorFlow.js WebGL Backend](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-webgl), WebGL backend for the browser.
* [TensorFlow.js WASM Backend](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm), WebAssembly backend for the browser.

#### Notes

* CPU backend is the default for running tests due to its higher numerical precision.
* CPU backend is the only supported backend for Node.js.
* WASM backend does not support all the ops and some test failures are thus expected.

#### Changing the backend

* When running tests in the browser, you can set the backend by passing a URL parameter `backend` that accept values `cpu`, `webgl` and `wasm`. e.g. [`?backend=webgl`](https://webmachinelearning.github.io/webnn-polyfill/test/?backend=webgl).
* When using the pre-built `webnn-polyfill.js` WebGL is the default backend. You can change the backend in your code as follows:

```js
    const backend = 'cpu';
    const tf = navigator.ml.createContext().tf;
    await tf.setBackend(backend);
    await tf.ready();
```
Please refer to the [`setPolyfillBackend()`](https://github.com/webmachinelearning/webnn-polyfill/search?q=setPolyfillBackend) usage in tests for concrete examples on how to best implement backend switching for your project.

## Usage

### Import the packages

#### Via NPM

```js
import '@webmachinelearning/webnn-polyfill';
```

#### Via a script tag

```html
<script src="https://cdn.jsdelivr.net/npm/@webmachinelearning/webnn-polyfill/dist/webnn-polyfill.js"></script>
```

### Samples

[Web Machine Learning Community Group](https://webmachinelearning.github.io/community/) provides various [Samples](https://webmachinelearning.github.io/webnn-samples-intro/) ([GitHub repo](https://github.com/webmachinelearning/webnn-samples)) that make use of the WebNN API. These samples fall back to the webnn-polyfill if the browser does not have a native implementation of the WebNN API available by default.

## Build and Test

### Setup

```sh
> git clone --recurse-submodules https://github.com/webmachinelearning/webnn-polyfill
> cd webnn-polyfill & npm install
```

### Build
#### Development build

```sh
> npm run build
```

#### Production build

```sh
> npm run build-production
```

### Test
#### Run tests in node.js.

```sh
> npm test
```

#### Run tests in web browser.

```sh
> npm start
```

Open the web browser and navigate to http://localhost:8080/test

#### Run only CTS tests in node.js.

```sh
> npm run test-cts
```

#### Run only CTS tests in web browser.

```sh
> npm start
```

Open the web browser and navigate to http://localhost:8080/test/cts.html

## Other scripts
### Build docs

```sh
> npm run build-docs
```

### Lint

```sh
> npm run lint
```

### Format

```sh
> npm run format
```

### Start dev server

```sh
> npm run dev
```

### Watch files

```sh
> npm run watch
```


## License

This project is licensed under the [Apache License Version 2.0](./LICENSE).
