# OpEn Agent

This package provides a classical, MPC based solution to the autonomous driving problem. We use [OpEn](https://alphaville.github.io/optimization-engine/), a fast, nonconvex optimizer to encode our MPC problem. OpEn generates a Rust solver to this problem which we then query to derive a trajectory to follow.

## Setup

Since we make use of Rust code-generation and compilation at run-time, you'll need to have Rust+Cargo installed.

The easiest way to get up and running is to follow the instructions at https://rustup.rs/

## Precompiled Binaries

The open_agent-X.XXX.whl files contained here are precompiled agent policies, these policy wheels can be rebuilt using `./setup.py` located in this same folder.

The purpose is for `./tests/test_open_agent.py` to have static policy binaries to test with.

Build the docs and see `docs/readme.rst:4.3.2 RLlib Example`.
