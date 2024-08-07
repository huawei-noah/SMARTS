.. _contributing:

Contributing
============

🎉 First and foremost, thanks for taking the time to contribute! 🎉

The following is a set of concise guidelines for contributing to SMARTS and its packages. Feel free to propose changes to this document in a pull request.

Code of conduct
---------------

+ Be respectful to other contributors. 
+ Keep criticism strictly to code when reviewing pull requests.
+ If in doubt about conduct ask the team lead or other contributors.

We encourage all forms of contributions to SMARTS, not limited to:

+ Code review and improvement
+ Community events
+ Blog posts and promotion of the project.
+ Feature requests
+ Patches
+ Test cases

Setting up your development environment
---------------------------------------

In addition to following the :ref:`setup` steps, you will need to install the ``[dev]`` dependencies.

.. code-block:: bash

    $ pip install -e .[dev]

Once done, you're all set to make your first contribution!

Where to get started
--------------------

Please take a look at the current open issues to see if any of them interest you.

Committing
----------

Please take care in using good commit messages as they're useful for debugging, reviewing code, and generally just shows care and quality. `How to Write a Git Commit Message <https://chris.beams.io/posts/git-commit/>`_ provides a good guideline. At a minimum,

1. Limit the subject line to 50 characters.
2. Capitalize the subject line.
3. Do not end the subject line with a period.
4. Use the imperative mood in the subject line.
5. If only changing documentation tag your commit with "[ci skip]".

Pre-Push Checklist
------------------

1. Do your best to see that your code compiles locally.
2. Do not push directly to ``master``. Instead make a branch and a pull request to the ``master`` branch.

Submission of a Pull Request
----------------------------

1. Rebase on master
2. Run ``make test`` locally to see if all test cases pass
3. If you change platform code, you are responsible to ensure all tests and all examples still run normally.   
4. Be sure to include new test cases if introducing a new feature or fixing a bug.
5. Update the documentation and apply comments to the public API. You are encouraged to add usage cases.
6. Update the ``CHANGELOG.md`` addressing what changes were made for the current version and make sure to indicate the PR # linking the changes.   
7. For PR description, describe the problem and add references to the related issues that the request addresses.
8. Request review of your code by at least two other contributors. Try to improve your code as much as possible to lessen the burden on others.
9. Do *not* keep long living branches. Branches are for a specific task. They should not become a sub repository.
10. After your PR gets approved by at least two other contributors you may merge your PR. 
11. Please enable squashing on your Pull Request before merging, this helps keep every commit on master in a working state and aids bisecting when searching for regressions.

In the body, give a reason for the pull request and tag in issues that the pull request solves. The ``WIP:`` is for pull requests that should raise discussion but are not in review state.

You are encouraged to review other people's pull requests and tag in relevant reviewers.

Code Format
-----------

The project follows a strict format requirement for python code. We made a decision early on in the project to use `Black <https://github.com/psf/black>`_. This makes formatting consistent while eliminating `bike shedding <https://bikeshed.com/>`_.
If you do not already have it please install it via ``pip install black``.

Formatting guarantees that your code will pass the CI formatting test case.

Documentation
-------------

All project code API, features, examples, etc, should be documented using `reStructuredText <https://www.sphinx-doc.org/>`_ at ``SMARTS/docs``. The docs are generated by Sphinx and hosted at `https://smarts.readthedocs.io/ <https://smarts.readthedocs.io/>`_.

All public python code interface should have `Google <https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings>`_ styled comments and ``"docstrings"``. Sphinx provides an `example <https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html>`_ Google style Python ``"docstrings"``.

Execute the following to build the docs locally.

.. code-block:: bash
    
    $ cd <path>/SMARTS
    $ pip install -e .[doc]
    $ make docs
    $ python3.8 -m http.server 8000 --bind 127.0.0.1 -d docs/_build/html
    # Open http://127.0.0.1:8000 in your browser

If documentation is incomplete please mark the area with a ``.. todo::`` as described in `sphinx.ext.todo <https://www.sphinx-doc.org/en/master/usage/extensions/todo.html>`_.

Feel free to further contribute to the documentation and look at :ref:`todo` for sections that may yet need to be filled.

Communication
-------------

Issues
^^^^^^

1. Always raise issues in GitHub. Verbal discussion and reports are helpful but *not* enough. Put things in writing please.
2. Raise specific, single-topic issues. If you find yourself having to use "and" in the issue title, you most likely want to create more than one.

Reporting Bugs
^^^^^^^^^^^^^^

Before reporting a bug please check the list of current issues to see if there are issues already open that match what you are experiencing.

When reporting a bug, include as much info as necessary for reproducing it. If you find a closed issue that appears to be the same problem you are experiencing; please open up a new issue referencing the original issue in the body of the new issue.

Tag the issue as a ``bug``.

Feature Requests
^^^^^^^^^^^^^^^^

Before requesting a feature please check the list of current issues to see if there is already a feature request similar to yours. Also, make sure that the feature you are requesting is not a bug. If it a bug see `Reporting Bugs`_.

Describe as best you can what the feature does and why it is useful. Visual aids help with understanding more complex features.

Tag the issue as a feature request using ``enhancement`` and if it takes more than a few lines to describe also tag with ``discussion``.

Generating Flame Graphs (Profiling)
-----------------------------------

Things inevitably become slow, when this happens, Flame Graph is a great tool to find hot spots in your code.

.. code-block:: bash

    $ cd <path>/SMARTS
    # python-flamegraph is needed to generate flamegraphs
    $ pip install git+https://github.com/asokoloski/python-flamegraph.git
    $ flamegraph_dir=./utils/third_party/tools
    $ mkdir -p flamegraph_dir
    $ curl https://raw.githubusercontent.com/brendangregg/FlameGraph/master/flamegraph.pl > ./utils/third_party/tools/flamegraph.pl
    $ chmod 777 {$flamegraph_dir}/flamegraph.pl
    $ make flamegraph scenario=./scenarios/sumo/loop script=./examples/e2_single_agent.py