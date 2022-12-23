# OpEn Agent

This package provides a classical, MPC based solution to the autonomous driving problem. We use [OpEn](https://alphaville.github.io/optimization-engine/), a fast, nonconvex optimizer to encode our MPC problem. OpEn generates a Rust solver to this problem which we then query to derive a trajectory to follow.

## Setup

Since we make use of Rust code-generation and compilation at run-time, you'll need to have Rust+Cargo installed.

The easiest way to get up and running is to follow the instructions at https://rustup.rs/

After that you may need to call `pip install . # with -e if preferred` up to 2 times. The first time will get the base dependencies, the second time will rebuild the solver. The solver will be output to the `../python-bindings-open-agent-solver` directory.