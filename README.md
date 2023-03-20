[![build and test](https://github.com/webmachinelearning/webnn-polyfill/workflows/build%20and%20test/badge.svg)](https://github.com/webmachinelearning/webnn-polyfill/actions)
[![deploy](https://github.com/webmachinelearning/webnn-polyfill/workflows/deploy/badge.svg)](https://github.com/webmachinelearning/webnn-polyfill/actions)

# WebNN Polyfill

A JavaScript implementation of the [Web Neural Network API](https://webmachinelearning.github.io/webnn/).

* [API docs](https://webmachinelearning.github.io/webnn-polyfill/docs/)
* [Unit tests](https://webmachinelearning.github.io/webnn-polyfill/test/)

## Backends

The implementation of this webnn-polyfill is based on [TensorFlow.js](https://github.com/tensorflow/tfjs) that supports the following 3 backends:

* [TensorFlow.js CPU Backend](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-cpu), pure-JS backend for Node.js and the browser.
* [TensorFlow.js WebGL Backend](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-webgl), WebGL backend for the browser.
* [TensorFlow.js WASM Backend](https://github.com/tensorflow/tfjs/blob/master/tfjs-backend-wasm), WebAssembly backend for the browser.

#### Notes

* CPU backend is the only supported backend for Node.js.
* WASM backend does not support all the ops and some test failures are thus expected.

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

### Set backend

WebNN Polyfill requires setting backend to enable TensorFlow.js.

* When running in Node.js, recommend using CPU backend for its higher numerical precision.
```js
    const backend = 'cpu';
    const context = await navigator.ml.createContext();
    const tf = context.tf;
    await tf.setBackend(backend);
    await tf.ready();
```

* When running in browsers, recommend using WebGL backend for better performance.

```js
    const backend = 'webgl'; // 'cpu' or 'wasm'
    const context = await navigator.ml.createContext();
    const tf = context.tf;
    await tf.setBackend(backend);
    await tf.ready();
```

* When running in browsers with WASM backend.

```js
    const backend = 'wasm';
    const context = await navigator.ml.createContext();
    const wasm = context.wasm;

    // 1- Enforce use Wasm SIMD binary
    wasm.setWasmPath(`${path}/tfjs-backend-wasm-simd.wasm`);

    // 2- Use Wasm SIMD + Threads bianry if supported both SIMD and Threads
    // 2.1- Configure by the path to the directory where the WASM binaries are located
    //        wasm.setWasmPaths(`https://unpkg.com/@tensorflow/tfjs-backend-wasm@${tf.version_core}/dist/`);
    //      or mapping from names of WASM binaries to custom full paths specifying the locations of those binaries
    //        wasm.setWasmPaths({
    //          'tfjs-backend-wasm.wasm': 'renamed.wasm',
    //          'tfjs-backend-wasm-simd.wasm': 'renamed-simd.wasm',
    //          'tfjs-backend-wasm-threaded-simd.wasm': 'renamed-threaded-simd.wasm'
    //        });
    wasm.setWasmPaths(${prefixOrFileMap}); 
    // 2.2- Configure threads number manually, or it will use the number of logical CPU cores as the threads count by default
    wasm.setThreadsCount(n); // n can be 1, 2, ...

    const tf = context.tf;
    await tf.setBackend(backend);
    await tf.ready();
```

Please refer to the [`setPolyfillBackend()`](https://github.com/webmachinelearning/webnn-polyfill/search?q=setPolyfillBackend) usage in tests for concrete examples on how to best implement backend switching for your project.

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

Default backend is CPU backend, you could change to use WebGL backend by `http://localhost:8080/test?backend=webgl`,<br>
or use Wasm backend by `http://localhost:8080/test?backend=wasm`

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
