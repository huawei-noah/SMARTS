import importlib
from pathlib import Path
from smarts.core.agent_interface import AgentInterface, RGB
from smarts.core.controllers import ActionSpaceType
from smarts.zoo.agent_spec import AgentSpec
from smarts.zoo.registry import register

def _verify_installation(pkg: str, module: str):
    try:
        lib = importlib.import_module(module, pkg)
    except (ModuleNotFoundError, ImportError):
        raise ModuleNotFoundError(
            "Zoo agent is not installed. "
            f"Install via `scl zoo install {str(Path(__file__).resolve().parent/pkg)}`."
        )

    return lib

def entry_point_dsac(**kwargs):
    pkg = "proximal_policy_optimization"
    module = ".policy"
    lib = _verify_installation(pkg=pkg, module=module)

    return AgentSpec(
        interface=AgentInterface(
            action=ActionSpaceType.TargetPose,
            top_down_rgb=RGB(
                width = 112,
                height = 112,
                resolution = 50 / 112,
            )
        ),
        agent_builder=lib.Policy,
    )

register(locator="proximal-policy-optimization-agent-v0", entry_point=entry_point_dsac)
