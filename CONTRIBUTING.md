# Contributing to SMARTS

We encourage all forms of contributions to SMARTS, not limited to:

* Code review and improvement
* Community events
* Blog posts and promotion of the project.
* Feature requests
* Patches
* Test cases

## Setting up your dev environment

In addition to following the setup steps in the README, you'll need to install the [dev] dependencies.

```bash
pip install -e .[dev]
```

Once done, you're all set to make your first contribution!

## Where to get started

Please take a look at the current open issues to see if any of them interest you. If you are unsure how to get started please take a look at the README.

## Submission of a Pull Request

1. Rebase on master
2. Run `make test` locally to see if all test cases pass
3. Be sure to include new test cases if introducing a new feature or fixing a bug.
4. Update the documentation and apply comments to the public API. You are encouraged to add usage cases.
5. Describe the problem and add references to the related issues that the request solves.
6. Request review of your code by other contributors. Try to improve your code as much as possible to lessen the burden on others.

## Formatting

We use `black` as our formatter. If you do not already have it please install it via `pip install black`.


## Generating Flame Graphs (Profiling)

Things inevitably become slow, when this happens, Flame Graph is a great tool to find hot spots in your code.

```bash
# You will need python-flamegraph to generate flamegraphs
pip install git+https://github.com/asokoloski/python-flamegraph.git

make flamegraph scenario=scenarios/loop script=examples/single_agent.py
```
