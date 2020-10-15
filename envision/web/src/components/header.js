import React, { useState } from "react";
import { Route, Switch } from "react-router-dom";
import { Layout, Select, Space, Button, Checkbox } from "antd";
import { LoadingOutlined } from "@ant-design/icons";
const { Option, OptGroup } = Select;
const { Header } = Layout;

export default function Header_({
  simulationIds,
  matchedSimulationId,
  onStartRecording,
  onStopRecording,
  onToggleShowScores,
  onToggleEgoView,
  onSelectSimulation,
}) {
  const [showScores, setShowScores] = useState(true);
  const [egoView, setEgoView] = useState(false);
  const [recording, setRecording] = useState(false);

  function toggleRecording() {
    let recording_ = !recording;
    if (recording_) {
      onStartRecording();
    } else {
      onStopRecording();
    }

    setRecording(recording_);
  }

  function toggleShowScores() {
    let showScores_ = !showScores;
    onToggleShowScores(showScores_);
    setShowScores(showScores_);
  }

  function toggleEgoView() {
    let egoView_ = !egoView;
    onToggleEgoView(egoView_);
    setEgoView(egoView_);
  }

  let selectValue = "";
  if (matchedSimulationId) {
    if (matchedSimulationId == "all") {
      selectValue = "all";
    } else {
      selectValue = `Simulation ${matchedSimulationId}`;
    }
  }

  return (
    <Header>
      <Space>
        <Select
          value={selectValue}
          style={{ width: 200 }}
          onChange={(value) => onSelectSimulation(value)}
        >
          <Option value="all">All Simulations</Option>
          <OptGroup label="Simulations">
            {simulationIds.map((id) => (
              <Option key={id} value={id}>{`Simulation ${id}`}</Option>
            ))}
          </OptGroup>
        </Select>

        <Switch>
          <Route exact={true} path="/all"></Route>
          <Route path="/:simulation">
            <Button
              type="primary"
              danger
              icon={recording ? <LoadingOutlined /> : null}
              onClick={toggleRecording}
            >
              Record
            </Button>
          </Route>
        </Switch>
      </Space>

      <Space style={{ float: "right" }}>
        <Checkbox defaultChecked onClick={toggleShowScores}>
          Show Scores
        </Checkbox>
        <Checkbox onClick={toggleEgoView}>Egocentric View</Checkbox>
      </Space>
    </Header>
  );
}
