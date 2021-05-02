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

const frameBufferModes = {
  NO_BIAS: 0, // randomly evict frames when buffer full
  PRIMACY_BIAS: 1, // prefer evicting more recent frames
  RECENCY_BIAS: 2, // more recent frames will have higher granularity
};

export default class Client {
  constructor({
    endpoint,
    delay = 2000,
    retries = Number.POSITIVE_INFINITY,
    maxFrameBufferSize = 300000,
    frameBufferMode = frameBufferModes.NO_BIAS,
  }) {
    this._endpoint = new URL(endpoint);
    this._wsEndpoint = new URL(endpoint);
    this._wsEndpoint.protocol = "ws";

    this._delay = delay;
    this._maxRetries = retries;
    this._glb_cache = {};

    this._maxFrameBufferSize = maxFrameBufferSize;
    this._frameBufferMode = frameBufferMode;

    this._sockets = {};
    this._stateQueues = {};
    this._simulationSelectedTime = {};
  }

  async fetchSimulationIds() {
    let url = new URL(this.endpoint);
    url.pathname = "simulations";
    let response = await fetch(url);
    if (!response.ok) {
      console.error("Unable to fetch simulation IDs");
      return [];
    } else {
      let data = await response.json();
      return data.simulations;
    }
  }

  seek(simulationId, seconds) {
    if (!(simulationId in this._sockets)) {
      this._sockets[simulationId] = null;
    }

    if (
      !this._sockets[simulationId] &&
      this._sockets[simulationId].readyState == WebSocket.OPEN
    ) {
      console.warn("Unable to seek because no connected socket exists");
      return;
    }

    this._sockets[simulationId].send(JSON.stringify({ seek: seconds }));
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
          let frames = JSON.parse(event.data);
          for (const frame of frames) {
            let state = JSON.parse(frame.state, (_, value) =>
              value === "NaN"
                ? Nan
                : value === "Infinity"
                ? Infinity
                : value === "-Infinity"
                ? -Infinity
                : value
            );
            if (
              stateQueue.length > 0 &&
              frame.current_elapsed_time <=
                stateQueue[stateQueue.length - 1].current_elapsed_time
            ) {
              // if it's moved back in time, it was from a seek and we're now
              // going to receive those frames again, so flush.
              stateQueue.length = 0;
            } else if (stateQueue.length > self._maxFrameBufferSize) {
              // the following is a placeholder to protect us
              // until we revisit the architecture, at which point
              // different policies can be implemented here.  for example,
              // we might eventually want this to depend on the playback mode,
              // or on events that happened in the simulation (although that
              // would require upstream support).  We might also want to
              // dump frames to a local file rather than just evicting them.
              switch (self._frameBufferMode) {
                case frameBufferModes.RECENCY_BIAS: {
                  // evenly thin out older frames to allow granular newer frames...
                  // (each time this is done, the earliest frames will get even thinner.)
                  // This allows for a "fast-forward-like catch up" to the most
                  // recent events in the simulation (when not in near-real-time
                  // playing mode).
                  stateQueue = stateQueue.filter((frame, ind) => ind % 2 == 0);
                  self._stateQueues[simulationId] = stateQueue;
                  break;
                }
                case frameBufferModes.PRIMACY_BIAS: {
                  // newer frames have a higher probability of being evicted...
                  let removeIndex = Math.floor(
                    stateQueue.length * Math.sqrt(Math.random())
                  );
                  stateQueue.splice(removeIndex, 1);
                  break;
                }
                case frameBufferModes.NO_BIAS:
                default: {
                  // randomly choose a frame to remove...
                  // spread the degradation randomly throughout the history.
                  let removeIndex = Math.floor(
                    stateQueue.length * Math.random()
                  );
                  stateQueue.splice(removeIndex, 1);
                }
              }
            }
            stateQueue.push({
              state: state,
              current_elapsed_time: frame.current_elapsed_time,
              total_elapsed_time: frame.total_elapsed_time,
            });
          }
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
    if (!(simulationId in this._sockets)) {
      this._sockets[simulationId] = null;
    }

    if (!(simulationId in this._stateQueues)) {
      this._stateQueues[simulationId] = [];
    }

    this._simulationSelectedTime[simulationId] = Date.now();
    let selectedTime = this._simulationSelectedTime[simulationId];

    while (true) {
      // If we dropped the connection or never connected in the first place
      let isConnected =
        this._sockets[simulationId] &&
        this._sockets[simulationId].readyState == WebSocket.OPEN;

      if (isConnected) {
        while (this._stateQueues[simulationId].length > 0) {
          // Removes the oldest element
          let item = this._stateQueues[simulationId].shift();
          let elapsed_times = [
            item.current_elapsed_time,
            item.total_elapsed_time,
          ];
          yield [item.state, elapsed_times];
        }
      } else {
        this._sockets[simulationId] = await this._obtainStream(
          simulationId,
          this._stateQueues[simulationId],
          this._maxRetries
        );
      }

      // This function can be triggered multiple times for the same simulation id
      // (i.e. everytime this simulation is selected from menu)
      // We only need to keep the most recent call to loop, all the previous calls can be returned
      if (selectedTime < this._simulationSelectedTime[simulationId]) {
        return;
      }

      // TODO: Make this "truly" async...
      await wait(1);
    }
  }

  get endpoint() {
    return this._endpoint;
  }
}
