import os
import subprocess


def test_sumo_lib():
    # import does runtime check by necessity
    from smarts.core.utils.sumo import sumolib


def test_sumo_version():
    from smarts.core.utils.sumo import traci, SUMO_PATH
    from smarts.core.utils import networking

    load_params = [
        "--start",
        "--quit-on-end",
        "--net-file=scenarios/loop/map.net.xml",
        "--no-step-log",
        "--no-warnings=1",
    ]

    sumo_port = networking.find_free_port()
    sumo_cmd = [
        os.path.join(SUMO_PATH, "bin", "sumo"),
        "--remote-port=%s" % sumo_port,
        *load_params,
    ]

    sumo_proc = subprocess.Popen(
        sumo_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )

    traci_conn = traci.connect(
        sumo_port, numRetries=10, proc=sumo_proc, waitBetweenRetries=0.1
    )

    assert (
        traci_conn.getVersion()[0] >= 20
    ), "TraCI API version must be >= 20 (SUMO 1.5.0)"
