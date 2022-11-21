'use strict';

/* eslint-disable no-undef */
importScripts('../dist/webnn-polyfill.js');

// Receive the message from the main thread
onmessage = (message) => {
  const context = navigator.ml.createContextSync();
  postMessage(context);
};
