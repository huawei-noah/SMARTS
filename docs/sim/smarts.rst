.. _smarts:

SMARTS Simulator
===============

The SMARTS class can be instantiated in the following way:

.. code-block:: python

  from smarts.core.smarts import SMARTS
  from smarts.core.agent_interface import AgentInterface, AgentType

  # Instantiate the simulator
  smarts: SMARTS = SMARTS(
    agent_interfaces={"A-007": AgentInterface.from_type(AgentType.Laner)},
    traffic_sim=None,
    envision=None if headless else Envision(),
  )

The step interface is similar to gym but with a few notable differences mainly due to the multi-agent nature of smarts.

.. code-block:: python

  from smarts.core.smarts import SMARTS
  # Instantiate the simulator
  obs = smarts.reset()
  dones = {"__all__": False}
  while not dones["__all__"]
    actions = {"A-007": "keep_lane"}
    obs, rewards, dones, extras = smarts.step(
      actions, # a dictionary of agent_id -> action pairs
      time_delta_since_last_step=0.1 # a variable
    )


The SMARTS simulator has the explict requirement to call `destroy()` before deleting the instance.

.. code-block:: python

  from smarts.core.smarts import SMARTSDestroyedError
  smarts = SMARTS()
  try:
    del smarts # Raises
  except SMARTSDestroyedError as e:
    print(e)

  smarts = SMARTS()
  ## Manual call to clean up SMARTS resources
  smarts.destroy()
  ## Now works
  del smarts

  smarts = SMARTS()
  try:
    exit() # Raises
  except SMARTSDestroyedError as e:
    print(e)

  smarts = SMARTS()
  # Program end # Asserts