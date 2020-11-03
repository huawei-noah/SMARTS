# Contributing to SMARTS

🎉 First and foremost, thanks for taking the time to contribute! 🎉

The following is a set of concise guidelines for contributing to SMARTS and its packages. Feel free to propose changes to this document in a pull request.

## Code of Conduct
- Be respectful. 
- Keep criticism strictly to code.
- Ask a project lead before revealing information about the project to outside entities. 
- If in doubt about conduct ask.

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
3. Do not push to `master`. Instead make a branch and a [merge request](#Merge-Requests)

## Merge Requests

1. Do _not_ keep long living branches. Branches are for a specific task. They should not become a sub repository.
2. Send in merge requests frequently. Do not send a large chunk of code in a single merge request. For large tasks, keep a plan, break things into stages and merge things in stage by stage
3. If you change platform code, you are responsible to ensure all tests and all examples still run normally.
4. Update the documentation and associated comments if you have made a change to the public api.
5. Please enable squashing on your Merge Request before merging, this helps keep every commit on master in a working state and aids bisecting when searching for regressions.


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

Formatting guarantees that your code will pass the CI formatting test case.

### Documentation(Format)
- Use [Markdown](https://daringfireball.net/projects/markdown/) liberally for in-repository documentation
- See [GitLab Markdown](https://docs.gitlab.com/ee/user/markdown.html#gitlab-markdown) for the flavor of in-repository Markdown.