.. _cli: 

Command Line Interface
======================

===
scl
===

Usage: scl [OPTIONS] COMMAND [ARGS]...

  The SMARTS command line interface. Use --help with each command for further
  information.

Options:
  --help  Show this message and exit.

Commands:
  envision  Commands to utilize an Envision server.
  run       Run an experiment on a scenario
  scenario  Generate, replay or clean scenarios.
  ultra     Utilites for working with the ULTRA benchmark.
  zoo       Build, install, or instantiate workers.

--------
envision
--------

Usage: scl envision [OPTIONS] COMMAND [ARGS]...

  Commands to utilize an Envision server. The Envision web server is used for
  visualization purposes. See `scl envision COMMAND --help` for further
  options.

Options:
  --help  Show this message and exit.

Commands:
  start  Start an Envision server.

start
^^^^^

Usage: scl envision start [OPTIONS]

  Start an Envision server.

Options:
  -p, --port INTEGER        Port Envision will run on.
  -s, --scenarios TEXT      A list of directories where scenarios are stored.
  -c, --max_capacity FLOAT  Max capacity in MB of Envision's playback buffer.
                            The larger the more contiguous history Envision
                            can store.
  --help                    Show this message and exit.

--------
scenario
--------

Usage: scl scenario [OPTIONS] COMMAND [ARGS]...

  Generate, replay or clean scenarios. See `scl scenario COMMAND --help` for
  further options.

Options:
  --help  Show this message and exit.

Commands:
  browse-waymo  Browse Waymo TFRecord datasets using...
  build         Generate a single scenario
  build-all     Generate all scenarios under the given directories
  clean         Remove previously generated scenario artifacts.
  replay        Play saved Envision data files in Envision.

build
^^^^^

Usage: scl scenario build [OPTIONS] <scenario>

  Generate a single scenario

Options:
  --clean             Clean previously generated artifacts first
  --allow-offset-map  Allows road network to be offset from the origin. If not
                      specified, creates a new network file if necessary.
  --help              Show this message and exit.

build-all
^^^^^^^^^

Usage: scl scenario build-all [OPTIONS] <scenarios>

  Generate all scenarios under the given directories

Options:
  --clean              Clean previously generated artifacts first
  --allow-offset-maps  Allows road networks (maps) to be offset from the
                       origin. If not specified, a new network file is created
                       if necessary.  Defaults to False except when there's
                       Traffic History data associated with the scenario.
  --help               Show this message and exit.

clean
^^^^^

Usage: scl scenario clean [OPTIONS] <scenario>

  Remove previously generated scenario artifacts.

Options:
  --help  Show this message and exit.

replay
^^^^^^

Usage: scl scenario replay [OPTIONS]

  Play saved Envision data files in Envision.

Options:
  -d, --directory TEXT
  -t, --timestep FLOAT  Timestep in seconds
  --endpoint TEXT
  --help                Show this message and exit.

browse-waymo
^^^^^^^^^^^^

Usage: scl scenario browse-waymo [OPTIONS] <script>

  Browse Waymo TFRecord datasets using smarts/waymo/waymo_browser.py, a text-
  based browser utility

Options:
  -t, --target-base-path PATH  Default target base path to export scenarios to
  -i, --import-tags PATH       .json file to import tags for tfRecord
                               scenarios from
  --help                       Show this message and exit.

-----
ultra
-----

Usage: scl ultra [OPTIONS] COMMAND [ARGS]...

  Utilites for working with the ULTRA benchmark.

Options:
  --help  Show this message and exit.

Commands:
  build  Build a policy

build
^^^^^

Usage: scl ultra build [OPTIONS] <policy>

  Build a policy

Options:
  --help  Show this message and exit.

---
zoo
---

Usage: scl zoo [OPTIONS] COMMAND [ARGS]...

  Build, install, or instantiate workers.

Options:
  --help  Show this message and exit.

Commands:
  build    Build a policy
  install  Attempt to install the specified agents from the given paths/url
  manager  Start the manager process which instantiates workers.

build
^^^^^

Usage: scl zoo build [OPTIONS] <policy>

  Build a policy

Options:
  --help  Show this message and exit.

manager
^^^^^^^

Usage: scl zoo manager [OPTIONS] [PORT]

  Start the manager process which instantiates workers. Workers execute remote
  agents.

Options:
  --help  Show this message and exit.

install
^^^^^^^

Usage: scl zoo install [OPTIONS] <script>

  Attempt to install the specified agents from the given paths/url

Options:
  --help  Show this message and exit.

---
run
---

Usage: scl run [OPTIONS] <script> [SCRIPT_ARGS]...

  Run an experiment on a scenario

Options:
  --envision                Start up Envision server at the specified port
                            when running an experiment
  -p, --envision_port TEXT  Port on which Envision will run.
  --help                    Show this message and exit.

