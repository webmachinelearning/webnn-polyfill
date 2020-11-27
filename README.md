[![build and test](https://github.com/webmachinelearning/webnn-polyfill/workflows/build%20and%20test/badge.svg)](https://github.com/webmachinelearning/webnn-polyfill/actions)
[![deploy](https://github.com/webmachinelearning/webnn-polyfill/workflows/deploy/badge.svg)](https://github.com/webmachinelearning/webnn-polyfill/actions)

# WebNN Polyfill

A JavaScript implementation of the [Web Neural Network API](https://webmachinelearning.github.io/webnn/).

* [API docs](https://webmachinelearning.github.io/webnn-polyfill/docs/)
* [Unit tests](https://webmachinelearning.github.io/webnn-polyfill/test/)

## Build and Test

### Setup

```sh
> npm install
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
> npm run cts
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
