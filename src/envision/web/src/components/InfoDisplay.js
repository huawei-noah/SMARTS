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

export default function InfoDisplay({
  data,
  ego_agent_ids,
  attrName,
  ego_only = false,
  data_formattter,
}) {
  return (
    <table style={{ margin: "15px", tableLayout: "auto" }}>
      <thead>
        <tr key="data-head">
          <th style={{ paddingRight: "15px" }}>{attrName}</th>
          <th>Agent</th>
        </tr>
      </thead>
      <tbody>
        {Object.entries(data).map(([id, score]) => {
          if (ego_only && !ego_agent_ids.includes(id)) {
            return null;
          }
          return (
            <tr key={`data-body-${id}`}>
              <td style={{ paddingRight: "15px" }}>{data_formattter(score)}</td>
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
