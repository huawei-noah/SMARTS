.. _bubbles:

How to work with Bubbles
========================

SMARTS provides the concept of a spatial-temporal bubble which allows for focused simulation interaction. Bubbles are intended to address the problem of scaling of interaction. Using resources globally results in wasted simulation resources if the most important behavior to an automous vehicle is in the nearby vicinity of that vehicle.

A bubble covers an area and filters traffic vehicles that pass through that zone. Vehicle pass into an buffer area of `shadowing` where an agent will begin observing from the vehicle, and then will fully take over control of that vehicle when it enters the bubble proper. SMARTS will replace traffic vehicles with the specified agents


Limtations
===========

There is current no policy for negotiating control between SMARTS and the traffic history. This issue is most apparent on releasing the simulated vehicle back to the history. Returning control naively results in a jump

\*\*This means that bubbles are currently incompatible with traffic history by issue of return of control to history when a vehicle exits the bubble.

Usage
=====

**Fixed bubbles**

Fixes a bubble to a static location defined either as an edge or a position.

.. code-block:: python
    import smarts.sstudio.types as t
    zoo_agent_actor = t.SocialAgentActor(
        # Unique agent name
        name="zoo-agent",
        # Formatted like (<python_module>\.*)+:[A-Za-z\-_]+(-v[0-9]+)?
        agent_locator=f"zoo.policies:zoo-agent-v0",
    )
    t.Bubble(
        # Edge snapped bubble
        zone=t.MapZone(start=("edge-west-WE", 0, 50), length=10, n_lanes=1),
        # Margin is an area where agents get observations but not control
        margin=2,
        # Agent actor information
        actor=zoo_agent_actor,
    ),

**Moving bubbles**

Bubbles that are vehicle relative can be attached to specific actors by specifying the id of the actor in the bubble.

.. code-block:: python
    import smarts.sstudio.types as t
    t.Bubble(
        ...,
        # The target of the bubble to follow the given actor
        follow_actor_id=t.Bubble.to_actor_id(laner_actor, mission_group="all"),
        # The offset from the target actor's vehicle(aligned with that vehicle's orientation)
        follow_offset=(-7, 10),
    ),

Dynamic Bubbles
===============
There is currently no interface for dynamic bubbles. However, if the `scenario` is exposed then the following is possible to define a bubble outside of `scenario studio``:

.. code-block:: python
    import smarts.sstudio.types as t
    scenario_iter = Scenario.scenario_variations(path)
    scenario = next(scenario)
    scenario.bubbles.append(
        t.Bubble(
            ...,
        ),
    )
    smarts.reset(scenario)



