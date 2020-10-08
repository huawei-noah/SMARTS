import regeneratorRuntime from "regenerator-runtime";
import React from "react";
import ReactDOM from "react-dom";
import { HashRouter as Router } from "react-router-dom";
import Client from "./client.js";
import App from "./components/app.js";
import "antd/dist/antd.dark.css";

let client = new Client({
  endpoint: "http://localhost:8081",
  delay: 2000,
  retries: 10,
});

ReactDOM.render(
  <Router>
    <App client={client} />
  </Router>,
  document.getElementById("root")
);
