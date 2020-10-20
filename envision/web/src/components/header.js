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
