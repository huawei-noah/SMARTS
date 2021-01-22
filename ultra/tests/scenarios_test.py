import unittest, shutil, os
from ultra.scenarios.generate_scenarios import *


class ScenariosTest(unittest.TestCase):
    def test_interface_generate(self):
        try:
            save_dir = "ultra/tests/scenarios/maps/easy/"
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.system(
                "python ultra/scenarios/interface.py generate --task 00 --level easy --root-dir ultra/tests/scenarios --save-dir ultra/tests/scenarios/maps/easy/map"
            )
            for dirpath, dirnames, files in os.walk(save_dir):
                if "traffic" in dirpath:
                    self.assertTrue("all.rou.xml" in files)
        except Exception as err:
            print(err)
            self.assertTrue(False)

    def test_generate_scenario(self):
        save_dir = "ultra/tests/scenarios/maps/easy/"
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

        build_scenarios(
            task="task00",
            level_name="easy",
            stopwatcher_behavior="aggressive",
            stopwatcher_route="south-west",
            root_path="ultra/tests/scenarios",
            save_dir=save_dir,
        )
        for dirpath, dirnames, files in os.walk(save_dir):
            if "traffic" in dirpath:
                self.assertTrue("all.rou.xml" in files)

    def test_generate_no_traffic(self):
        save_dir = "ultra/tests/scenarios/maps/no-traffic/"
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        build_scenarios(
            task="task00",
            level_name="no-traffic",
            stopwatcher_behavior="aggressive",
            stopwatcher_route="south-west",
            root_path="ultra/tests/scenarios",
            save_dir=save_dir,
        )
        for dirpath, dirnames, files in os.walk(save_dir):
            if "traffic" in dirpath:
                self.assertTrue("all.rou.xml" not in files)
