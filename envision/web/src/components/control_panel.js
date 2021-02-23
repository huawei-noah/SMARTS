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
import { Tree } from "antd";

export const attrs = Object.freeze({
  score: 0,
  speed: 1,
  position: 2,
  heading: 3,
  laneID: 4,
});

export const agentModes = Object.freeze({
  egoObs: "5",
  socialObs: 6,
});

const treeData = [
  {
    title: "Vehicle Observation",
    key: "Vehicle Observation",
    children: [
      {
        title: "Ego Agent Observation",
        key: agentModes.egoObs,
        children: [
          {
            title: "score",
            key: attrs.score,
          },
          {
            title: "speed",
            key: attrs.speed,
          },
          {
            title: "position",
            key: attrs.position,
          },
          {
            title: "heading",
            key: attrs.heading,
          },
          {
            title: "lane id",
            key: attrs.laneID,
          },
        ],
      },
      {
        title: "Inclucdes Social Agents",
        key: agentModes.socialObs,
      },
    ],
  },
];

export default function ControlPanel({ showControls, toggleControlModes }) {
  const [expandedKeys, setExpandedKeys] = useState([agentModes.egoObs]);
  const [checkedKeys, setCheckedKeys] = useState([
    attrs.score,
    agentModes.socialObs,
  ]);
  const [autoExpandParent, setAutoExpandParent] = useState(true);

  const onExpand = (expandedKeys) => {
    // if not set autoExpandParent to false, if children expanded, parent can not collapse.
    // or, you can remove all expanded children keys.
    setExpandedKeys(expandedKeys);
    setAutoExpandParent(false);
  };

  const onCheck = (checkedKeys, info) => {
    setCheckedKeys(checkedKeys);
    toggleControlModes({ [info.node.key]: info.checked });
  };

  const onSelect = (selectedKeys, info) => {
    if (checkedKeys.includes(info.node.key)) {
      // remove from list
      setCheckedKeys((prevKeys) =>
        prevKeys.filter((key) => key != info.node.key)
      );
      toggleControlModes({ [info.node.key]: false });
    } else {
      // add to list
      setCheckedKeys((prevKeys) => [...prevKeys, info.node.key]);
      toggleControlModes({ [info.node.key]: true });
    }
  };

  return (
    <div
      style={{
        zIndex: "1",
        position: "relative",
        display: "flex",
        top: "0",
        left: "0",
        maxWidth: "50%",
        paddingRight: "3px",
      }}
    >
      {showControls ? (
        <Tree
          checkable
          onExpand={onExpand}
          expandedKeys={expandedKeys}
          autoExpandParent={autoExpandParent}
          onCheck={onCheck}
          checkedKeys={checkedKeys}
          onSelect={onSelect}
          treeData={treeData}
        />
      ) : null}
    </div>
  );
}
