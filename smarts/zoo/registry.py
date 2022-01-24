# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from smarts.core.utils.class_factory import ClassRegister

agent_registry = ClassRegister()


def register(locator: str, entry_point, **kwargs):
    """Register an AgentSpec with the zoo.

    In order to load a registered AgentSpec it needs to be reachable from a
    directory contained in the PYTHONPATH.

    Args:
        locator:
            A string in the format of 'locator-name'
        entry_point:
            A callable that returns an AgentSpec or an AgentSpec object

    For example:

    .. code-block:: python

        register(
            locator="motion-planner-agent-v0",
            entry_point=lambda **kwargs: AgentSpec(
                interface=AgentInterface(waypoints=True, action=ActionSpaceType.TargetPose),
                agent_builder=MotionPlannerAgent,
            ),
        )
    """

    agent_registry.register(locator=locator, entry_point=entry_point, **kwargs)


def make(locator: str, **kwargs):
    """Create an AgentSpec from the given locator.

    In order to load a registered AgentSpec it needs to be reachable from a
    directory contained in the PYTHONPATH.

    Args:
        locator:
            A string in the format of 'path.to.file:locator-name' where the path
            is in the form `{PYTHONPATH}[n]/path/to/file.py`
        kwargs:
            Additional arguments to be passed to the constructed class.
    """

    from smarts.core.agent import AgentSpec

    agent_spec = agent_registry.make(locator, **kwargs)
    assert isinstance(
        agent_spec, AgentSpec
    ), f"Expected make to produce an instance of AgentSpec, got: {agent_spec}"

    return agent_spec
