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
