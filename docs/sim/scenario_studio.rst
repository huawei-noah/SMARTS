.. _scenario_studio:

===============
Scenario Studio
===============

The Scenario Studio of SMARTS is a stand alone package ``sstudio`` that supports flexible and expressive scenario specification. 

At the most basic level scenarios combine a map with a traffic description.  SMARTS interacts with such maps abstractly (polymorphically) through the ``RoadMap`` interface in order to support multiple map formats and allow for extending to support new formats.

The ``sstudio`` domain specific language (DSL) has a simple ontology defined by :mod:`smarts.sstudio.types` to help express your scenario needs in SMARTS.

SMARTS ships with a variety of pre-designed scenarios, which can be found in ``SMARTS/scenarios`` and ``SMARTS/smarts/scenarios`` directories.

Creating scenarios
==================

Workflow to create a scenario is as follows.

1. Create a new scenario folder, preferably under ``SMARTS/scenarios`` directory, and create a new ``scenario.py`` file in the new folder.
2. Design the map.
   
   * For a SUMO road network, create the map using `netedit <https://sumo.dlr.de/docs/NETEDIT.html>`_ and save it in the new folder.
   * For other road networks, define a ``MapSpec`` object in ``scenario.py``.

3. To add social vehicle, traffic, or agent missions, define a traffic spec in ``scenario.py``.
4. Finally, build the scenario by running

   .. code-block:: bash

      $ cd <path>/SMARTS
      $ scl scenario build-all <path>/<new_scenario_folder>

Following sections below explain how to handle and edit maps, traffic, social agents, agent missions, and any required additional packages, in the ``scenario.py`` file.

Generate traffic
================

.. code-block:: python

    traffic_actor = TrafficActor(name="car", speed=Distribution(sigma=0.2, mean=0.8),)

    # Add 10 social vehicles with random routes.
    traffic = Traffic(
        engine="SUMO",
        flows=[
            # Generate flows that lasts for 10 hours.
            Flow(
                route=RandomRoute(), 
                begin=0, 
                end=10 * 60 * 60, 
                rate=25, 
                actors={traffic_actor: 1},
            )
            for i in range(10)
        ]
    )

The ``engine`` argument to ``Traffic`` can either be ``"SUMO"`` or ``"SMARTS"``. 
Defaults to ``engine="SUMO"``. 
``engine="SUMO"`` can only be used on SUMO road networks. For other map types use ``engine="SMARTS"``.

``traffic_actor`` is used as a spec for traffic actors (e.g. Vehicles, Pedestrians, etc). The defaults provided are for a car.
Acceleration, deceleration, speed distribution, imperfection distribution, and other configs, can be specified for the traffic.
For more ``TrafficActor`` config see :mod:`smarts.sstudio.types`.

``Flow`` is used to generate repeated vehicle runs on the same route. Vehicle route and departure rate can be configured here.

After ``traffic`` is supplied to the ``gen_scenario`` function, a dir named "traffic" will be created under ``output_dir`` which contains background vehicle and route definitions.

This a short ``scenario.py`` example of how it works.

.. literalinclude:: ./minimal_scenario_studio.py
   :language: python

Simply run the ``scenario.py`` file as a regular Python script to generate the scenario.

.. code-block:: bash

    $ python3.8 scenarios/scenario.py

.. important::
    If you want to train a model on one scenario, remember to set the ``end`` time of ``Flow`` larger or equal to your expected training time, since SMARTS will continue the flow after each ``reset`` call. However, if there are multiple scenarios to train
    for one worker, you can relax this restriction since after the scenario change, the flow will also be reset to the beginning time.

Generate missions
=================

Scenario Studio also allows generation of *missions* for ego agents and social agents. These missions are similar to routes for social vehicles. When we run ``gen_scenario``, ``missions.rou.xml`` file will be created in the output dir.

.. code-block:: python

    missions = [
        Mission(
            Route(
                begin=("edge0", 0, "random"), 
                end=("edge1", 0, "max"),
            ),
        ),
    ]

Generate friction map
=====================

The Scenario Studio of SMARTS also allows the generation of a *friction map* which consists of a list of *surface patches* for ego agents and social agents. These surface patches are using PositionalZone as in the case of bubbles. When we run ``gen_scenario`` passing in ``friction_maps``, a "friction_map.pkl" file will be created under the output dir:

.. code-block:: python

  friction_maps = [
    RoadSurfacePatch(
        PositionalZone(pos=(153, -100), size=(2000, 6000)),
        begin_time=0,
        end_time=20,
        friction_coefficient=0.5,
    ),
  ]


Generate road map
=================

SMARTS was initially designed to use maps in the SUMO road network format; it supports these natively.
However as of v0.5, SMARTS now supports other custom map formats, as
long as a class that implements the ``smarts.core.RoadMap`` interface is provided to read these.
An example implementation of the [OpenDRIVE map format](https://www.asam.net/index.php?eID=dumpFile&t=f&f=4089&token=deea5d707e2d0edeeb4fccd544a973de4bc46a09)
is provided as ``smarts.core.OpenDriveRoadNetwork`` to show how this can be done.
Also see ``smarts.core.WaymoMap`` for an example of support for a format from Waymo.

Create a custom map
-------------------
If not utilizing a built-in map type (i.e., Sumo or OpenDRIVE road networks or a Waymo map),
define a ``MapSpec`` object in your ``scenario.py``.

.. code-block:: python

  # function that can create custom map objects using a map_spec
  def custom_map_builder(map_spec):
     # ...
     return map_object, map_hash

  map_spec = MapSpec(source="path_or_uri", builder_fn=custom_map_builder)


Convert an existing map to SUMO
-------------------------------
If you have a suitable file in another format, you can turn it into a SUMO road network using the ``sumo2mesh.py`` conversion utility:

.. code-block:: bash

  python3 -m smarts.sstudio.sumo2mesh dataset_public/2lane_sharp/map.net.xml dataset_public/2lane_sharp/map.glb --format=glb
  python3 -m smarts.sstudio.sumo2mesh dataset_public/2lane_sharp/map.net.xml dataset_public/2lane_sharp/map.egg --format=egg


Create a SUMO map
-----------------

You can edit your own SUMO map through [SUMO's NETEDIT](https://sumo.dlr.de/docs/NETEDIT.html) and export it in a map.net.xml format.
First, to start ``netedit``, run the following on terminal:

.. code-block:: bash

  netedit

On the top left bar, "file" -> "new network" to create a new map.

Use shortcut key "e" to change to edge mode. Click "chain" and "two-way" icons located on the far right of top tool bar, shown in the image below:

.. image:: ../_static/chain_two_way.png

Then click on map to start creating new edges and lanes.

Note that SMARTS prefers to have "internal links" (connections) as part of any Junctions.  You can enable these by 
going to "Processing" -> "Options", choosing the "Junctions" section, and then making sure the 
"no-internal-links" checkbox is *unchecked*.


Edit an existing SUMO map
-------------------------

"file" -> "Open Network..." to open an existing map.

Click on the inspect icon to enable inspect mode

.. image:: ../_static/inspect.png

Click on any edge to inspect detail and modify properties.

.. image:: ../_static/map_lane.png

The selected block is an edge with id "gneE72". It contains 3 lanes with lane index 0, 1, 2.

To modify the properties, for example change the number of lanes to 2 lanes by changing 3 to 2 on the "numLanes" field, and press
"enter" to make the change. Then press "ctrl+s" to save. Finally, make sure to rebuild the scenario.

.. code-block:: bash

  scl scenario build --clean <path-to-scenario-folder>

To create custom connections between edges, first click the following icon on top bar:

.. image:: ../_static/connection_icon.png

The first lane you select would be the source lane, highlighted in blue. Then select other lanes as target lanes to connect to.

.. image:: ../_static/create_connection.png



Create traffic routes
=====================

For example, using the following ``Route`` definition:

.. code-block:: python

  Route(begin=("gneE72", 0, "random"), end=("edge2", 1, "max"),)

``begin=("gneE72", 0, "random")`` defines the route to start on edge with id ``gneE72`` and at lane index ``0``,
which is the same lane as the selected lane in the figure above. ``"random"`` here specifies the amount of offset on the lane to start the route.



Additional Packages
-------------------

Scenarios can reference remote packages or local zoo agent packages by including a ``requirements.txt`` file in the root of the scenario folder. These additional packages will be installed during the scenario build.

In the requirements.txt file:
.. code-block::

    --extra-index-url http://localhost:8080
    <dependency>==1.0.0
    rl-agent==1.0.0
    ...

Then in the scenario.py file:
.. code-block:: python

    t.SocialAgentActor(
        name="my-rl-agent",
        agent_locator="rl_agent:rl_agent-v1"
    )








.. note::

    The "correctness" of traffic and missions is partially your responsibility. Specifically, ensuring that the start positions of ego vehicle mission routes and social vehicle traffic routes don't overlap is not handled by ``sstudio``. If they were to overlap a collision will be immediately detected and the episode will end.