// Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
const wait = (ms) => {
  return new Promise((resolve) => setTimeout(resolve, ms));
};

export default class Client {
  constructor({ endpoint, delay = 2000, retries = Number.POSITIVE_INFINITY }) {
    this._endpoint = new URL(endpoint);
    this._wsEndpoint = new URL(endpoint);
    this._wsEndpoint.protocol = "ws";

    this._delay = delay;
    this._maxRetries = retries;
    this._glb_cache = {};
  }

  async fetchSimulationIds() {
    let url = new URL(this.endpoint);
    url.pathname = "simulations";
    let response = await fetch(url);
    if (!response.ok) {
      console.error("Unable to fetch simulation IDs.");
      return [];
    } else {
      let data = await response.json();
      return data.simulations;
    }
  }

  async _obtainStream(simulationId, stateQueue, remainingRetries) {
    let self = this;
    let url = new URL(self._wsEndpoint);
    url.pathname = `simulations/${simulationId}/state`;

    try {
      return await new Promise((resolve, reject) => {
        let socket = new WebSocket(url);
        socket.onopen = (event) => {
          console.debug("Socket connected!");
          resolve(socket);
        };

        socket.onclose = (event) => {
          console.debug("Socket closed");
        };

        socket.onmessage = (event) => {
          let data = JSON.parse(event.data, (_, value) =>
            value === "NaN"
              ? Nan
              : value === "Infinity"
              ? Infinity
              : value === "-Infinity"
              ? -Infinity
              : value
          );
          stateQueue.push(data);
        };

        socket.onerror = (error) => {
          console.warn(
            `Socket encountered error=${error.message} ` +
              `trying to connect to endpoint=${url}`
          );
          reject(error);
        };
        return socket;
      });
    } catch (error) {
      if (remainingRetries === 0) throw error;
      console.info(
        `Retrying connection, attempts remaining=${remainingRetries}`
      );

      remainingRetries -= 1;
      await wait(self._delay);
      return await self._obtainStream(
        simulationId,
        stateQueue,
        remainingRetries
      );
    }
  }

  async *worldstate(simulationId) {
    let socket = null;
    let stateQueue = [];

    while (true) {
      // If we dropped the connection or never connected in the first place
      let isConnected = socket && socket.readyState === WebSocket.OPEN;

      if (isConnected) {
        while (stateQueue.length > 0) {
          yield stateQueue.pop();
        }
      } else {
        socket = await this._obtainStream(
          simulationId,
          stateQueue,
          this._maxRetries
        );
      }

      // TODO: Make this "truly" async...
      await wait(1);
    }
  }

  get endpoint() {
    return this._endpoint;
  }
}
