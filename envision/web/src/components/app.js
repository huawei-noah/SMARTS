import React, { useState, useEffect, useRef } from "react";
import {
  Route,
  Switch,
  useHistory,
  withRouter,
  useRouteMatch,
} from "react-router-dom";
import ReactDOM from "react-dom";
import html2canvas from "html2canvas";
import { RecordRTCPromisesHandler, invokeSaveAsDialog } from "recordrtc";
import { Layout } from "antd";
const { Content } = Layout;

import Header from "./header.js";
import Simulation from "./simulation.js";
import SimulationGroup from "./simulation_group.js";

// To fix https://git.io/JftW9
window.html2canvas = html2canvas;

function App(props) {
  const [simulationIds, setSimulationIds] = useState([]);
  const [showScores, setShowScores] = useState(true);
  const simulationCanvasRef = useRef(null);
  const recorderRef = useRef(null);
  const history = useHistory();

  // also includes all
  const routeMatch = useRouteMatch("/:simulation");
  const matchedSimulationId = routeMatch ? routeMatch.params.simulation : null;

  useEffect(() => {
    (async () => {
      let ids = await props.client.fetchSimulationIds();
      if (ids.length > 0) {
        if (!matchedSimulationId || !ids.includes(matchedSimulationId)) {
          history.push(`/${ids[0]}`);
        }
      }

      setSimulationIds(ids);
    })();
  }, []);

  async function onStartRecording() {
    recorderRef.current = new RecordRTCPromisesHandler(
      simulationCanvasRef.current,
      {
        type: "canvas",
      }
    );
    await recorderRef.current.startRecording();
  }

  async function onStopRecording() {
    await recorderRef.current.stopRecording();
    let blob = await recorderRef.current.getBlob();
    invokeSaveAsDialog(blob, `envision-${Math.round(Date.now() / 1000)}.webm`);
  }

  function onSelectSimulation(simulationId) {
    history.push(`/${simulationId}`);
  }

  return (
    <Layout className="layout" style={{ width: "100%", height: "100%" }}>
      <Header
        simulationIds={simulationIds}
        matchedSimulationId={matchedSimulationId}
        onSelectSimulation={onSelectSimulation}
        onStartRecording={onStartRecording}
        onStopRecording={onStopRecording}
        onToggleShowScores={(show) => setShowScores(show)}
      />
      <Content>
        <Switch>
          <Route exact={true} path="/all">
            <SimulationGroup
              client={props.client}
              simulationIds={simulationIds}
              showScores={showScores}
            />
          </Route>
          <Route
            path="/:simulation"
            render={() => {
              return (
                <Simulation
                  canvasRef={simulationCanvasRef}
                  client={props.client}
                  simulationId={matchedSimulationId}
                  showScores={showScores}
                />
              );
            }}
          />
        </Switch>
      </Content>
    </Layout>
  );
}

export default withRouter(App);
