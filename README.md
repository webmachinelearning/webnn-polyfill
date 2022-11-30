[![build and test](https://github.com/webmachinelearning/webnn-polyfill/workflows/build%20and%20test/badge.svg)](https://github.com/webmachinelearning/webnn-polyfill/actions)
[![deploy](https://github.com/webmachinelearning/webnn-polyfill/workflows/deploy/badge.svg)](https://github.com/webmachinelearning/webnn-polyfill/actions)

# WebNN Polyfill

A JavaScript implementation of the [Web Neural Network API](https://webmachinelearning.github.io/webnn/).

* [API docs](https://webmachinelearning.github.io/webnn-polyfill/docs/)
* [Unit tests](https://webmachinelearning.github.io/webnn-polyfill/test/)

## Backends

The implementation of this webnn-polyfill is based on [TensorFlow.js](https://github.com/tensorflow/tfjs) that supports following 4 backends.

* [TensorFlow.js CPU Backend](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-cpu), pure-JS backend for Node.js and the browser.
* [TensorFlow.js WebGL Backend](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-webgl), WebGL backend for the browser.
* [TensorFlow.js WASM Backend](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm), WebAssembly backend for the browser.

If not set, tests under the webnn-polyfill use CPU as default backend, as which has higher numerical precision than other backends. Tests may fail under WASM backend as some ops have not been implemented/supported in WASM backend.

* For node test, we only support CPU backend.
* For browser test, you can set backend by passing a URL parameter: `backend`, it accepts `cpu`, `webgl` and `wasm`. e.g. `?backend=webgl`.

If not set, the built `webnn-polyfill.js` uses WebGL as default backend, you can set backend by referring to following code snippet:

```js
    const backend = 'cpu';
    const tf = navigator.ml.createContext().tf;
    await tf.setBackend(backend);
    await tf.ready();
```

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

WebNN provides various [Samples](https://github.com/webmachinelearning/webnn-samples) built with WebNN API, which would use WebNN Polyfill on browsers where WebNN API is not implemented yet.

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

This project is following [Apache License Version 2.0](./LICENSE).
