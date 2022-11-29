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
// FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
import React from "react";
import Simulation from "./simulation.js";
import { Button } from "antd";

const SimulationGroup = ({
  simulationIds,
  client,
  controlModes,
  playingMode,
  egoView,
  onSelectSimulation,
}) => {
  return (
    <div style={{ padding: "10px" }}>
      {simulationIds.map((simId) => (
        <Button
          title="click the simulation to maximize"
          onClick={(value) => onSelectSimulation((value = simId))}
          key={simId}
          style={{
            float: "left",
            margin: "10px",
            width: "480px",
            height: "480px",
          }}
        >
          <Simulation
            client={client}
            simulationId={simId}
            egoView={egoView}
            controlModes={controlModes}
            playingMode={playingMode}
          />
        </Button>
      ))}
    </div>
  );
};

export default SimulationGroup;
