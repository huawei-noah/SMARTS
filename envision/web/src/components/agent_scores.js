import React from "react";

export default function AgentScores({ scores, style }) {
  return (
    <table style={{ margin: "15px", tableLayout: "auto", ...style }}>
      <thead>
        <tr key="scores-head">
          <th style={{ paddingRight: "15px" }}>Score</th>
          <th>Agent</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(scores).map(([id, score]) => {
          return (
            <tr key={`scores-body-${id}`}>
              <td style={{ paddingRight: "15px" }}>
                {parseFloat(score).toFixed(2)}
              </td>
              <td
                style={{
                  maxWidth: "400px",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {id}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
