import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from smarts.sstudio.sumo2mesh import generate_glb_from_sumo_network


@contextmanager
def temp_scenario(name: str, map: str):
    with tempfile.TemporaryDirectory() as temp_dir:
        scenario = Path(temp_dir) / name
        scenario.mkdir()

        test_maps_dir = Path(__file__).parent.parent
        shutil.copyfile(test_maps_dir / map, scenario / "map.net.xml")
        generate_glb_from_sumo_network(
            str(scenario / "map.net.xml"), str(scenario / "map.glb")
        )

        yield scenario
