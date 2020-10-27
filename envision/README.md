# Envision

Envision is a visualization front-end for SMARTS providing real-time view of environment state. It's built on web-technologies (including [React](https://reactjs.org/), [WebGL](https://www.khronos.org/webgl/), and [websockets](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API)) allowing it to run easily in your browser. Envision is composed of a few parts: a client which SMARTS uses directly; a server used for state broadcasting; and the web application where all the visualization and rendering happens.

![](../docs/_static/smarts_envision.gif)

## Running

```bash
# From SMARTS project root; starts SMARTS and the Envision server
# ...if you want to change the startup command for SMARTS update the supervisord.conf file
supervisord

# Then to visit the Envision web app in your browser,
http://localhost:8081/
```

## Development

To contribute to envision it's easiest to start and control the processes manually. Start the Envision server by running,

```bash
# From SMARTS project root; runs on port 8081 by default
python envision/server.py
```

Then start the Envision web application. npm (version >= 6) and node (version >= 12) are required.

```bash
cd envision/web

# Install dependencies
npm install

# Build, run dev server, and watch code changes
npm start
```

## Deployment

If you've made changes to the Envision web application you'll want to save an updated distribution which users access directly (so they don't have to setup Envision's development dependencies). Simply run,

```bash
# Saves to envision/web/dist
npm run build
```

## Extras

### Data Recording and Replay

For recording simply add `envision_record_data_replay_path` to the `gym.make(...)` call,

```python
env = gym.make(
    "smarts.env:hiway-v0",
    scenarios=args.scenarios,
    agents={AGENT_ID: agent},
    headless=args.headless,
    visdom=False,
    timestep_sec=0.1,
    envision_record_data_replay_path="./data_replay",
)
```

then run with `supervisord` (currently Envision server needs to be up for data recording to work).

For replay make sure you have Envision server running then use the following tool - passing in your replay files,

```bash
scl scenario replay -d ./data_replay/1590892375a -t 0.1

INFO:root:Replaying 1 record(s) at path=data_replay/1590892375a with timestep=0.1s
```
