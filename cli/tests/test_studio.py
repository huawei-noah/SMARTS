import shutil
import os
import tempfile
from xml.etree.ElementTree import ElementTree


def _compare_files(file1, file2):
    with open(file1) as f:
        items = [x.items() for x in ElementTree(file=f).iter()]

    with open(file2) as f:
        generated_items = [x.items() for x in ElementTree(file=f).iter()]

    sorted_items = sorted(items)
    sorted_generated_items = sorted(generated_items)
    if not sorted_items == sorted_generated_items:
        for a, b in zip(sorted_items, sorted_generated_items):
            assert a == b, f"{file1} is different than {file2}"


def test_scenario_generation_unchanged():
    number_of_comparisons_greater_than_0 = False
    with tempfile.TemporaryDirectory() as temp_dir1:
        with tempfile.TemporaryDirectory() as temp_dir2:
            loc1 = temp_dir1 + "/scenarios"
            loc2 = temp_dir2 + "/scenarios"
            from cli.studio import _build_all_scenarios
            import re

            _hashseed = os.getenv("PYTHONHASHSEED")
            assert _hashseed not in (None, "random"), f"PYTHONHASHSEED is {_hashseed}"

            shutil.copytree("scenarios/sumo", loc1)
            _build_all_scenarios(True, True, [loc1], 42)

            shutil.copytree("scenarios/sumo", loc2)
            _build_all_scenarios(True, True, [loc2], 42)

            for dirpath, dirnames, files in os.walk(loc1):
                if "traffic" in dirpath:
                    assert len(files) > 0
                    for file in files:
                        dir2 = re.sub(loc1, loc2, dirpath)
                        _compare_files(dirpath + "/" + file, dir2 + "/" + file)
                        number_of_comparisons_greater_than_0 = True

    assert number_of_comparisons_greater_than_0
