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
import React, { useState, useEffect, useRef } from "react";
import {
  Route,
  Switch,
  useHistory,
  withRouter,
  useRouteMatch,
} from "react-router-dom";
import html2canvas from "html2canvas";
import { RecordRTCPromisesHandler, invokeSaveAsDialog } from "recordrtc";
import { Layout } from "antd";
const { Content } = Layout;
import Header from "./header";
import Simulation from "./simulation";
import SimulationGroup from "./simulation_group";
import { attrs, agentModes } from "./control_panel";
import PlaybackBar from "./playback_bar";
import ControlPanel from "./control_panel.js";
import { useToasts } from "react-toast-notifications";
import transcode from "../helpers/transcode";

// To fix https://git.io/JftW9
window.html2canvas = html2canvas;

function App({ client }) {
  const [simulationIds, setSimulationIds] = useState([]);
  const [showControls, setShowControls] = useState(true);
  const [controlModes, setControlModes] = useState({
    [attrs.score]: true,
    [attrs.speed]: false,
    [attrs.position]: false,
    [attrs.heading]: false,
    [attrs.laneId]: false,
    [agentModes.socialObs]: true,
  });
  const [egoView, setEgoView] = useState(false);
  const [currentElapsedTime, setCurrentElapsedTime] = useState(0);
  const [totalElapsedTime, setTotalElapsedTime] = useState(1);
  const [playing, setPlaying] = useState(true);
  const simulationCanvasRef = useRef(null);
  const recorderRef = useRef(null);
  const { addToast } = useToasts();
  const history = useHistory();

  // also includes all
  const routeMatch = useRouteMatch("/:simulation");
  const matchedSimulationId = routeMatch ? routeMatch.params.simulation : null;

  useEffect(() => {
    const fetchRunningSim = async () => {
      let ids = await client.fetchSimulationIds();
      if (ids.length > 0) {
        if (!matchedSimulationId || !ids.includes(matchedSimulationId)) {
          history.push(`/${ids[ids.length - 1]}`);
        }
      }
      setSimulationIds(ids);
    };

    // checks if there is new simulation running every 3 seconds.
    const interval = setInterval(fetchRunningSim, 3000);
    return () => clearInterval(interval);
  }, []);

  async function onStartRecording() {
    recorderRef.current = new RecordRTCPromisesHandler(
      simulationCanvasRef.current,
      {
        type: "canvas",
        mimeType: "video/webm;codecs=h264",
      }
    );
    await recorderRef.current.startRecording();
  }

  async function onStopRecording() {
    addToast("Stopping recording", { appearance: "info" });
    await recorderRef.current.stopRecording();

    let onMessage = (message) => addToast(message, { appearance: "info" });
    let blob = await recorderRef.current.getBlob();
    let outputBlob = await transcode(blob, onMessage);
    invokeSaveAsDialog(
      outputBlob,
      `envision-${Math.round(Date.now() / 1000)}.mp4`
    );
  }

  function onSelectSimulation(simulationId) {
    history.push(`/${simulationId}`);
  }

  function toggleControlModes(attr) {
    setControlModes((prevMode) => ({
      ...prevMode,
      ...attr,
    }));
  }

  return (
    <Layout className="layout" style={{ width: "100%", height: "100%" }}>
      <Header
        simulationIds={simulationIds}
        matchedSimulationId={matchedSimulationId}
        onSelectSimulation={onSelectSimulation}
        onStartRecording={onStartRecording}
        onStopRecording={onStopRecording}
        onToggleShowControls={(show) => setShowControls(show)}
        onToggleEgoView={(view) => setEgoView(view)}
      />
      <Content>
        <Switch>
          <Route exact={true} path="/all">
            <SimulationGroup
              client={client}
              simulationIds={simulationIds}
              showControls={showControls}
              egoView={egoView}
            />
          </Route>
          <Route
            path="/:simulation"
            render={() => {
              return (
                <div
                  style={{
                    display: "flex",
                    width: "100%",
                    height: "100%",
                    flexDirection: "column",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      flex: 1,
                      flexDirection: "row",
                    }}
                  >
                    <ControlPanel
                      showControls={showControls}
                      toggleControlModes={toggleControlModes}
                    />
                    <Simulation
                      canvasRef={simulationCanvasRef}
                      client={client}
                      simulationId={matchedSimulationId}
                      showControls={showControls}
                      controlModes={controlModes}
                      egoView={egoView}
                      onElapsedTimesChanged={(current, total) => {
                        setCurrentElapsedTime(current);
                        setTotalElapsedTime(total);
                      }}
                      style={{ flex: "1" }}
                      playing={playing}
                    />
                  </div>
                  <PlaybackBar
                    currentTime={currentElapsedTime}
                    totalTime={totalElapsedTime}
                    onSeek={(seconds) => {
                      setCurrentElapsedTime(seconds);
                      client.seek(matchedSimulationId, seconds);
                    }}
                    style={{ height: "80px" }}
                    playing={playing}
                    setPlaying={setPlaying}
                  />
                </div>
              );
            }}
          />
        </Switch>
      </Content>
    </Layout>
  );
}

export default withRouter(App);
