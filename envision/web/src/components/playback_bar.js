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
import { Slider } from "antd";
import { PauseCircleOutlined, PlayCircleOutlined } from "@ant-design/icons";

const buttonStyle = {
  fontSize: "24px",
};

export default function PlaybackBar({
  onSeek,
  currentTime = 0,
  totalTime = 1,
  playing = true,
  setPlaying,
}) {
  const [_, setCurrentTime] = useState(currentTime);

  return (
    <div
      style={{
        padding: "10px 20px",
        background: "#1F1F1F",
        display: "flex",
        flexDirection: "row",
      }}
    >
      <button
        onClick={() => setPlaying(!playing)}
        style={{
          backgroundColor: "Transparent",
          border: "none",
          outline: "none",
        }}
      >
        {playing ? (
          <PauseCircleOutlined style={buttonStyle} />
        ) : (
          <PlayCircleOutlined style={buttonStyle} />
        )}
      </button>
      <Slider
        style={{ flex: 1 }}
        value={currentTime}
        max={totalTime}
        step={0.0001}
        onChange={(seconds) => {
          setCurrentTime(seconds);
          onSeek(seconds);
        }}
        tipFormatter={(tip) => `${tip.toFixed(2)}s`}
        tooltipVisible
      />
    </div>
  );
}
