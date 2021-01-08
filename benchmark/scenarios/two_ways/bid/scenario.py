from pathlib import Path

from smarts.sstudio import gen_scenario
import smarts.sstudio.types as t

missions = [
    t.Mission(t.Route(begin=("-gneE1", 0, 5), end=("-gneE1", 0, 97))),
    t.Mission(t.Route(begin=("gneE1", 0, 5), end=("gneE1", 0, 97))),
    t.Mission(t.Route(begin=("-gneE1", 0, 20), end=("-gneE1", 0, 97))),
    t.Mission(t.Route(begin=("gneE1", 0, 20), end=("gneE1", 0, 97))),
]

gen_scenario(
    t.Scenario(ego_missions=missions), output_dir=Path(__file__).parent,
)
