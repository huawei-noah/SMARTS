Quickstart: Designing a Simple Experiment
=========================================

First we'll need to define what our agent looks like, we'll then place the agent in a scenario and run through a few episodes of simulation.

Specifying the Agent
--------------------

The agent is defined in terms of the interface it expects from the environment and the responses an agent produces. To help bridge the gap between the environment and your agent, we also introduce adapters.

:class:`smarts.core.agent_interface.AgentInterface`
   This is where you can control the interface between SMARTS and your agent.

:class:`smarts.core.agent.Agent`
   This is the brains of the agent, you will need to implement the interface defined by :class:`smarts.core.agent.Agent` in order to give the agent some behaviour.

Adapters:
  Adapters bridge the gab between SMARTS and your agent. It is sometimes useful to preprocess the input and outputs of an agent, we won't be needing adapter for this little walkthrough but for a more in-depth treatment see :ref:`adapters`.


AgentInterface :class:`smarts.core.agent_interface.AgentInterface`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we choose the interface between SMARTS and your agent. Select which sensors to enable on your vehicle and the action space for you agent.

Some common configurations have been packaged up under :class:`smarts.core.agent_interface.AgentType` and can be instantiated via

.. code-block:: python

   from smarts.core.agent_interface import AgentInterface, AgentType

   AgentInterface.from_type(AgentType.Tracker)

This `AgentType.Tracker` preset gives us :class:`smarts.core.agent_interface.Waypoints` and the trajectory following action space `ActionSpaceType.Trajectory`, see :class:`smarts.core.controllers.ActionSpaceType` for more available action spaces.


Agent :class:`smarts.core.agent.Agent`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next up, we need to define the behaviour of the agent. That is to say, we want to map the observations from the sensors we configured above to the action space we chose.

This is done by implementing the :class:`smarts.core.agent.Agent` interface:

.. code-block:: python

   from smarts.core.bezier_motion_planner import BezierMotionPlanner
   from smarts.core.agent import Agent

   class ExampleAgent(Agent):
       def __init__(self, target_speed = 10):
           self.motion_planner = BezierMotionPlanner()
           self.target_speed = target_speed

       def act(self, obs):
           ego = obs.ego_vehicle_state
           current_pose = np.array([*ego.position[:2], ego.heading])

           # lookahead (at most) 10 waypoints
           target_wp = obs.waypoint_paths[0][:10][-1]
           dist_to_wp = target_wp.dist_to(obs.ego_vehicle_state.position)
           target_time = dist_to_wp / self.target_speed

           # Here we've computed the pose we want to hold given our target
           # speed and the distance to the target waypoint.
           target_pose_at_t = np.array(
               [*target_wp.pos, target_wp.heading, target_time]
           )

           # The generated motion planner trajectory is compatible
           # with the `ActionSpaceType.Trajectory`
           traj = self.motion_planner.trajectory(
               current_pose, target_pose_at_t, n=10, dt=0.5
           )
           return traj

Here we are implementing a simple lane following agent using the BezierMotionPlanner. The `obs` argument to `ExampleAgent.act()` will contain the observations specified in the `AgentInterface` above, and it's expected that the return value of the `act` method matches the `ActionSpaceType` chosen as well. (This constraint is relaxed when adapters are introduced.)


AgentSpec :class:`smarts.core.agent.AgentSpec`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These pieces are brought together by the :class:`smarts.core.agent.AgentSpec`:

.. code-block:: python

   agent_spec = AgentSpec(
       interface=AgentInterface.from_type(AgentType.Tracker)
       # params are passed to the agent_builder when we build the agent
       agent_params={"target_speed": 5},
       agent_builder=ExampleAgent
   )

The :class:`smarts.core.agent.AgentSpec` acts as a container to store the information we need to build an agent, we can distribute this spec safely between process' to aid in parallelism and once we have it in the right spot, we can instantiate the :class:`smarts.core.agent.Agent` with

.. code-block:: python

   agent = agent_spec.build_agent()

Putting it all together
-----------------------

We can run this agent with "scenarios/loop", one of the scenarios packaged with SMARTS using the familiar gym interface:

.. code-block:: python

   import gym
   from smarts.core.agent import AgentSpec, Agent
   from smarts.core.agent_interface import AgentInterface, AgentType
   from smarts.core.bezier_motion_planner import BezierMotionPlanner
   from smarts.core.utils.episodes import episodes

   class ExampleAgent(Agent):
       def __init__(self, target_speed = 10):
           self.motion_planner = BezierMotionPlanner()
           self.target_speed = target_speed

       def act(self, obs):
           ego = obs.ego_vehicle_state
           current_pose = np.array([*ego.position[:2], ego.heading])

           # lookahead (at most) 10 waypoints
           target_wp = obs.waypoint_paths[0][:10][-1]
           dist_to_wp = target_wp.dist_to(obs.ego_vehicle_state.position)
           target_time = dist_to_wp / self.target_speed

           # Here we've computed the pose we want to hold given our target
           # speed and the distance to the target waypoint.
           target_pose_at_t = np.array(
               [*target_wp.pos, target_wp.heading, target_time]
           )

           # The generated motion planner trajectory is compatible
           # with the `ActionSpaceType.Trajectory`
           traj = self.motion_planner.trajectory(
               current_pose, target_pose_at_t, n=10, dt=0.5
           )
           return traj

   AGENT_ID = "Agent-007"
   agent_spec = AgentSpec(
       interface=AgentInterface.from_type(AgentType.Tracker)
       agent_params={"target_speed": 5},
       agent_builder=ExampleAgent
   )

   env = gym.make(
       "smarts.env:hiway-v0",
       scenarios=["scenarios/loop"],
       agent_specs={AGENT_ID: agent_spec},
   )

   for episode in episodes(n=100):
       agent = agent_spec.build_agent()
       observations = env.reset()
       episode.record_scenario(env.scenario_log)

       dones = {"__all__": False}
       while not dones["__all__"]:
           agent_obs = observations[AGENT_ID]
           action = agent.act(agent_obs)
           observations, rewards, dones, infos = env.step({AGENT_ID: action})
           episode.record_step(observations, rewards, dones, infos)

   env.close()

The scenario is deterministic in totality. This means that assuming all agents take the exact same 
actions the entire scenario will play back deterministically but each episode will have different
behaviour.
