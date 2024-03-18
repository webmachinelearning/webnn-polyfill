'use strict';

/* eslint-disable no-undef */
importScripts('../dist/webnn-polyfill.js');

// Receive the message from the main thread
onmessage = async (message) => {
  if (message) {
    const shape = message.data[0];
    const inputs = message.data[1];
    const outputs = message.data[2];
    const context = await navigator.ml.createContext();
    const builder = new MLGraphBuilder(context);
    const x = builder.input('x', {dataType: 'float32', dimensions: shape});
    const y = builder.relu(x);
    const graph = await builder.build({y});
    const result = await context.compute(graph, inputs, outputs);
    postMessage(result.outputs);
  }
};
