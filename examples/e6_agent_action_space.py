"""This is an example to show how agent interface action formatting configuration works."""
from pathlib import Path

from examples.tools.argument_parser import empty_parser
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import Controllers
from smarts.core.controllers.action_space_type import ActionSpaceType
from smarts.env.utils.action_conversion import ActionOptions, get_formatters


def display_spaces():
    tn, _, _ = f"{ActionOptions=}".partition("=")
    action_formatters = get_formatters()
    for name, action_type in ActionSpaceType.__members__.items():
        unformatted_types, unformatted_schema_names = Controllers.get_action_shape(
            action_type
        )
        formatted = action_formatters[action_type]
        print(f"======= {name} =======")
        print(f"- For {ActionOptions.unformatted!r} -")
        print(f"{unformatted_types = !r}")
        print(f"{unformatted_schema_names = !r}")
        print()
        print(
            f"- For {tn} {(ActionOptions.multi_agent, ActionOptions.full, ActionOptions.default)} -"
        )
        print(f"Gym space = {formatted.space!r}")
        print()


def display_agent_type_spaces():
    method_name, _, _ = f"{AgentInterface.from_type=}".partition("=")
    print(
        f"Note that `{method_name}` generates a pre-configured agent type with an existing action space."
    )

    for name, agent_type in AgentType.__members__.items():
        print(f"{name.ljust(30)}: {AgentInterface.from_type(agent_type).action}")


def main(*_, **kwargs):
    display_spaces()
    display_agent_type_spaces()


if __name__ == "__main__":
    parser = empty_parser(Path(__file__).stem)
    args = parser.parse_args()
    main()
