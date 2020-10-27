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
