'use strict';

/* eslint-disable no-undef */
importScripts('../dist/webnn-polyfill.js');

// Receive the message from the main thread
onmessage = (message) => {
  if (message) {
    const shape = message.data[0];
    const inputs = message.data[1];
    const outputs = message.data[2];
    const context = navigator.ml.createContextSync();
    const builder = new MLGraphBuilder(context);
    const x = builder.input('x', {type: 'float32', dimensions: shape});
    const y = builder.relu(x);
    const graph = builder.buildSync({y});
    context.computeSync(graph, inputs, outputs);
    postMessage(outputs);
  }
};
