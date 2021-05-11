# Curriculum learning inside ULTRA

Curriculum learning describes a type of learning in which the agent progressively trains with easier tasks first and then gradually increases difficulty of those tasks. The outcome will be that the agent's learning performance/efficiency will increase. A quick example is the education curriculum in schools that provide a guide to "teachers" as to what is essential for learning, so that the "students" are exposed to robust academic experiences. In contrast to reinforcement learning, the rl agents are like students who can utilize past experiences/knowledge to complete the present task. 

You might wonder that ULTRA itself is defined to be a single task problem, solving the unprotected left-turn. However, agents in ULTRA learn from scratch which means gaining experiences in the basics of driving, centering in lanes, stopping behind other vehicles and on different types of traffic density. Then after we can focus on the target task (unprotected left-turn). Instead of presenting randomly selected scenarios to an agent, we can construct a curriculum which prioritizes the agent to first pass the easier scenarios.

Curricula can be either created before hand or on-the-fly. In ULTRA, we refer to curricula which are built prior to training as "static" and curricula that are not fully established or can change during training are referred as "dynamic".

**Static curricula** in ULTRA currently use an implementation called the grade-based (GB) structure. The grade-based structure allows scenarios to be sequenced/ordered during training and testing runs. This is done by dividing an experiment into so called "grades", which are combinations of different tasks and levels. These grades are basically buckets for combinations of tasks and their levels.

When an agent is put into training with curriculum-mode enabled (`ultra/train.py` is run with flag `--curriculum-mode True`), a coordinator object is initialized. It is responsible for setting up the scenarios in each grade and performing "graduation": switching of the agent into the next grade.

Unlike the regular training setup (when curriculum mode is disabled), there is another task folder called `curriculum/` under `ultra/scenarios/` which has the blueprint of which set of scenarios are in each grade:

```yaml
curriculum:
  static:
    toggle: <bool> # Turn on static curriculum settings
    grades:
      1: [[<task>,<level>]]
      2: [[<task>,<level>], [<task>,<level>]]
      # 3: [[<task>,<level>], ... ,[<task>,<level>]] can have as many combinations of tasks and levels
    conditions:
      eval_per_grade: <bool> # Evaluation takes places after completion of grade. An "exam" at the end of year
      episode_based: # Agent graduates after completing a N number of episodes in each grade
        toggle: <bool> # Enable the condition
        cycle: <bool> # Option to keep cycling through grades to episodes limit
      pass_based: # Agent graduates after getting an average completion rate, the average is taken over the eval-rate (sampling-rate)
        toggle: <bool> # Enable the condition
        pass_rate: <float> # Scalar between 0 and 1; describes the threshold completion rate (%)
        sample_rate: <int> # Samples for some parameter (average_reached_goal) at every N episodes
        warmup_episodes: <int> # Starting N episodes where sampling does not occur, thus no graduation can take place
```
A more specific example in which we take the three levels from task 1 and distribute it among three grades:

```yaml
curriculum:
  static:
    toggle: True
    grades:
      1: [[1,no-traffic]]
      2: [[1,easy]]
      3: [[1,hard]]
      conditions:
        eval_per_grade: True
        episode_based:
          toggle: True 
          cycle: True 
        pass_based:
          toggle: False 
          pass_rate: 0.50 # If the average scenario passed exceeds more than 0.5 then grade is switched
          sample_rate: 30 # Every 30 episodes slot, the average reached goal (arg) is calculated (arg = total_scenario_passed / sample_rate)
          warmup_episodes: 100 # Starting at any grade, the first 100 episodes will be subjected to no sampling
```
Now when we enter grade 1, the agent will only face the scenarios that are part of (task 1, level no-traffic). Grade 2 consists of scenarios that art part of (task 1, level easy), and grade 3 has scenarios from (task 1, level hard). The conditions for graduation to the next grade is that the agent will complete N number of episodes in each grade. N is the quotient of the total number of episodes / total number of grades. The other condition is that agents will only graduate from a grade if it has reached a specified "passing mark". The "passing mark" can be any metric (`env_score`, `reached_goal`, `collisions`) that is averaged over N sample of episodes (which is defined in the curriculum config file).

To use the grade-based structure, you will follow the same steps as any other training with the exception that you will need to turn on curriculum-mode by setting `--curriculum-mode True`.

**Dynamic curricula** in ULTRA do not adhere to any structure. The work on developing dynamic curricula is still in progress. Although, the tools that will be used in dynamic curricula have already been implemented in ULTRA. In non-curriculum mode or in static curricula we use levels to define which scenarios will be used in either the entire experiment or a specific grade. However, dynamic curricula will be able to generate and add scenarios at every episode or at an episodic sampling rate (e.g. every 100 episodes).

Scenarios can be abstracted over traffic densities, intersection shapes, and speeds. This means that the levels that contain definitions of scenarios do not need to be very general. We mean that levels can be used to include a definition of **a single scenario**. Here is an example:

- Level easy contains different types of densities and speeds (also applicable to test scenarios):

```yaml
  easy: # (general level)
    train:
      # ... irrelevant info is left out
      intersection_types:
        2lane_t:
           percent: 1.0
           specs: [[50kmh,low-density,0.21],[70kmh,low-density,0.20],[100kmh,low-density,0.20], #61%
                   [50kmh,mid-density,0.11],[70kmh,mid-density,0.11],[100kmh,mid-density,0.11], #33%,
                   [50kmh,no-traffic,0.01],[70kmh,no-traffic,0.01],[100kmh,no-traffic,0.01], #3%
                   [50kmh,high-density,0.01],[70kmh,high-density,0.01],[100kmh,high-density,0.01]] # 3%
```

- However, a level can also be very specific (2_lane t intersection with only medium density traffic that flows @ 70kmh). The following level will only produce scenarios with this specification:

```yaml
  2t-mid-70: # 2lane_t, mid-density, 70kmh (specific level)
    train:
      # ... irrelevant info is left out
      intersection_types:
        2lane_t:
           percent: 1.0
           specs: [[70kmh,mid-density,1]] # 100%
```

Due to work in progress with further research into curriculum learning. The dynamic curriculum implementation is separated from the static curriculum. In future updates a more blended architecture will be implemented. For now to use the dynamic curriculum, we must update a few params in the curriculum config file:

```yaml
  curriculum:
    static:
      # N/A
    dynamic:
      toggle: <bool> # Turn on dynamic curriculum settings
      tasks_levels_used: <list> # All the tasks & levels that can be used
      sampling_rate: <list> # The rate at which the curriculum is updated in terms of episodes
```
A specific example with values to describe the dynamic curriculum setup:

```yaml
  curriculum:
    static:
      toggle: False
      # N/A
    dynamic:
      toggle: True
      tasks_levels_used: [["1","no-traffic"], ["1","low-density"], ["1","mid-density"], ["1","high-density"]]
      sampling_rate: 50
```
We can use this method to create these single-scenario levels that only generate a specific range of scenarios. So if the user wants to introduce low-density traffic (with any speed) at a given episode or timestep they can "pull" (build) scenarios from that level and use them. Although, an overlying algorithm or policy will automate the process of building and introducing scenarios to the agents.

Scenarios that have been dynamically created will be stored in a staging directory. The directory will be flushed depending on when the "teacher" or user needs to update the curriculum.

For testing scenarios, the same process applies. However, test scenarios will be only generated once in bulk. You can either specify a test level with desired settings and the numbers of scenarios to be generated in the task configs, or specify the number of scenarios and type of scenarios on-the-fly as well.

- Execute `ultra/train.py`. The following is a list of available arguments for curriculum learning only. To view full list of arguments please refer to [Train and Evaluate a Baseline Agent](docs/getting_started.md).
  - `--curriculum-mode`: Use curriculum learning, will ignore the tasks and levels flag (default is False).
  - `--curriculum-dir`: Path to the curriculum directory which is used to gather task and level information of the grades (default is ../scenarios/curriculum/).
  - `--curriculum-build-scenarios`: Option to automatically build all the scenarios (available for static curricula) which will be needed from each grade. If you have already build the scenarios then simply ignore this flag (default is False).
  - `--curriculum-scenarios-root-dir`: Specify the directory where the curriculum tasks (config files) are stored (default is `ultra/scenarios/`).
  - `--curriculum-scenarios-save-dir`: Specify the directory to save the scenarios in. Default is to save the scenarios inside its respective task directory (default is None).

> The base tools which are needed for implementing curriculum learning in ULTRA are complete, however with continuous progress on the research front, these tools will be subject to change.
