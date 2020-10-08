import React from "react";
import Simulation from "./simulation.js";

const SimulationGroup = ({ simulationIds, showScores, client }) => {
  return (
    <div style={{ padding: "10px" }}>
      {simulationIds.map((simId) => (
        <div
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
            showScores={showScores}
          />
        </div>
      ))}
    </div>
  );
};

export default SimulationGroup;
