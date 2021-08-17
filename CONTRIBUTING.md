# Contributing to SMARTS
🎉 First and foremost, thanks for taking the time to contribute! 🎉

The following is a set of concise guidelines for contributing to SMARTS and its packages. Feel free to propose changes to this document in a pull request.

## Code of Conduct
- Be respectful to other contributors. 
- Keep criticism strictly to code when reviewing pull requests.
- If in doubt about conduct ask the team lead or other contributors.

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

## Committing

Please take care in using good commit messages as they're useful for debugging, reviewing code, and generally just shows care and quality. [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/) provides a good guideline. At a minimum,

1. Limit the subject line to 50 characters
2. Capitalize the subject line
3. Do not end the subject line with a period
4. Use the imperative mood in the subject line
5. If only changing documentation tag your commit with `[ci skip]`

## Pre-Push Checklist

1. Do your best to see that your code compiles locally.
2. Run `make format`. See [Formatting](#Formatting).
3. Do not push to `master`. Instead make a branch and a [pull request](#Pull-Requests)

## Submission of a Pull Request

1. Rebase on master
2. Run `make test` locally to see if all test cases pass
3. If you change platform code, you are responsible to ensure all tests and all examples still run normally.   
4. Be sure to include new test cases if introducing a new feature or fixing a bug.
5. Update the documentation and apply comments to the public API. You are encouraged to add usage cases.
6. Update the `CHANGELOG.md` addressing what changes were made for the current version and make sure to indicate the PR # linking the changes.   
7. For PR description, describe the problem and add references to the related issues that the request addresses.
8. Request review of your code by at least two other contributors. Try to improve your code as much as possible to lessen the burden on others.
9. Do _not_ keep long living branches. Branches are for a specific task. They should not become a sub repository.
10. After your PR gets approved by at least other contributors you may merge your PR. 
11. Please enable squashing on your Pull Request before merging, this helps keep every commit on master in a working state and aids bisecting when searching for regressions.


In the body, give a reason for the pull request and tag in issues that the pull request solves. The `WIP:` is for pull requests that should raise discussion but are not in review state.

You are encouraged to review other people's pull requests and tag in relevant reviewers.

## Communication

### Issues

1. Always raise issues in GitLab. Verbal discussion and reports are helpful but _not_ enough. Put things in writing please.
2. Raise specific, single-topic issues. If you find yourself having to use "and" in the issue title, you most likely want to create more than one.

### Reporting Bugs
Before reporting a bug please check the list of current issues to see if there are issues already open that match what you are experiencing.

When reporting a bug, include as much info as necessary for reproducing it. If you find a closed issue that appears to be the same problem you are experiencing; please open up a new issue referencing the original issue in the body of the new issue.

Tag the issue as a `bug`.

### Feature Requests
Before requesting a feature please check the list of current issues to see if there is already a feature request similar to yours. Also, make sure that the feature you are requesting is not a bug. If it a bug see [Reporting Bugs](Reporting-Bugs).

Describe as best you can what the feature does and why it is useful. Visual aids help with understanding more complex features.

Tag the issue as a feature request using `enhancement` and if it takes more than a few lines to describe also tag with `discussion`.

## Formatting

### Python(Format)

1. Always run `make format` before committing.

The project follows a strict format requirement for python code. We made a decision early on in the project to use [Black](https://github.com/psf/black). This makes formatting consistent while eliminating [bike shedding](http://bikeshed.com/).
If you do not already have it please install it via `pip install black`.

Formatting guarantees that your code will pass the CI formatting test case.

### Documentation(Format)
- Use [Markdown](https://daringfireball.net/projects/markdown/) liberally for in-repository documentation
- See [GitLab Markdown](https://docs.gitlab.com/ee/user/markdown.html#gitlab-markdown) for the flavor of in-repository Markdown.



## Generating Flame Graphs (Profiling)

Things inevitably become slow, when this happens, Flame Graph is a great tool to find hot spots in your code.

```bash
# You will need python-flamegraph to generate flamegraphs
pip install git+https://github.com/asokoloski/python-flamegraph.git

make flamegraph scenario=scenarios/loop script=examples/single_agent.py
```
