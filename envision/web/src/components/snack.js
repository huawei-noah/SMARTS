// MIT License
//
// Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

const snackStates = {
  entering: { transform: "translate3d(0, 120%, 0) scale(0.9)" },
  entered: { transform: "translate3d(0, 0, 0) scale(1)" },
  exiting: { transform: "translate3d(0, 120%, 0) scale(0.9)" },
  exited: { transform: "translate3d(0, 120%, 0) scale(0.9)" },
};

export default function Snack({
  children,
  transitionDuration,
  transitionState,
}) {
  return (
    <div
      style={{
        alignItems: "center",
        backgroundColor: "rgba(0, 0, 0, 0.75)",
        borderRadius: "5px",
        color: "white",
        display: "flex",
        flexWrap: "wrap",
        justifyContent: "space-between",
        marginBottom: "8px",
        minWidth: "200",
        maxWidth: "500",
        padding: "6px 24px",
        pointerEvents: "initial",
        transitionProperty: `transform`,
        transitionDuration: `${transitionDuration}ms`,
        transitionTimingFunction: `cubic-bezier(0.2, 0, 0, 1)`,
        transformOrigin: "bottom",
        zIndex: 2,
        ...snackStates[transitionState],
      }}
    >
      <div style={{ fontSize: "10pt", padding: "8px 0" }}>{children}</div>
    </div>
  );
}
