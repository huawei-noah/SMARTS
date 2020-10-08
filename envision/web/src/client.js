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
