'use strict';

/* eslint-disable no-undef */
importScripts('../dist/webnn-polyfill.js');

// Receive the message from the main thread
onmessage = async (message) => {
  const context = await navigator.ml.createContext();
  postMessage(context);
};
