# Curriculum learning inside ULTRA

Curriculum learning describes a type of learning in which the agent progressively trains with easier tasks at first and then gradually increases difficulty of tasks. The outcome will be that the agent's learning performance/efficiency will increase. A quick example is the education curriculum in schools that provide a guide to "teachers" as to what is essential for learning, so that the "students" are exposed to robust academic experiences. In contrast to reinforcement learning, the rl agents are like students who utilizes past experiences/knowledge to complete the present task. 

You might wonder that ULTRA itself is defined to be a single task problem, solving the unprotected left-turn. However, agents in ULTRA learn from scratch which means gaining experience on the basics of driving, centering in lanes, stopping behind other vehicles and on different types of traffic density, then after we can focus on the target task (unprotected left-turn). Instead of presenting randomly selected scenarios to an agent, we can construct a curriculum which prioritizes the agent to first pass the easier scenarios. 

Curriculum can be either created before hand or on-the-fly. In ULTRA, we refer to curriculas which are built prior to training as static and curriculas that are not fully established or can change during training are referred as dynamic. 

**Static curriculas** in ULTRA currently uses an implementation called the grade-based (GB) structure. The grade-based structure allows scenarios to be sequenced/ordered during training/testing runs. This is done by dividing an experiment into so called grades, which are combinations of different tasks and levels. These grades are basically buckets for certain types of tasks and there levels.

When an agent is put into training with curriculum-mode enable (run train.py with flag --curriculum-mode True), an coordinator object is initialized. It is responsible for setting up the scenarios in each grade and performing graduation (switching) of the agent into the next grade.

Unlike the regular the training setup (curriculum mode disable), there is another task folder called curriculum under ultra/scenarios which has the blueprint of which set of scenarios are in each grade

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
A more specific example in which we take the three levels from task 1 and distribute it among three grades

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
Now when we enter grade 1, the agent will only face the scenarios that are part of (task 1, level easy), same applies to grades 2 and 3. The condition for graduation is that the agent will complete N number of episodes in each grade. N is the quotient of the total number of episodes / total number of grades. 

To use the grade-based structure, you will follow the same steps as any other training with exception that you will need to turn on --curriculum-mode by setting that flag to True

**Dynamic curriculas** in ULTRA does not adhere to any structure. In fact work is still in progress to create a on-the-fly curriculum. Although, the tools that will be used in dynamic curriculas have been implemented in ULTRA. In non-curriculum mode or in static curriculas we used to levels to define which scenarios will be used in either the entire experiment or a specific grade. However, dynamic curriculas are going to be able to have to access to generate and add scenarios at every episodes or at an episodic sampling rate (eg. every 100 episodes). 

Scenarios can be abstracted over traffic densities, intersection shapes and/or speeds. This means that the levels that contain definitions of scenarios does not need to be very general. We mean that levels can be used to include a definition of **an single scenario**. Here is an example.

- Level easy contains different types of densities and speeds (also applicable to test parameter)

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

- However a level can be very specific (2_lane t intersection with only medium density that flows @ 70kmh). This level will only produce scenarios with that specification

```yaml
  2t-mid-70: # 2lane_t, mid-density, 70kmh (specific level)
    train:
      # ... irrelevant info is left out
      intersection_types:
        2lane_t:
           percent: 1.0
           specs: [[70kmh,mid-density,1]] # 100%
```

Due to work in progress with further research into curriculum learning. The dynamic curriculum implementation is separated from the static curriculum. In future updates a more blended architecture will be implemented. For now to use the dynamic curriculum, we must update a few params curriculum config file 

```yaml
  curriculum:
    static:
      # N/A
    dynamic:
      toggle: <bool> # Turn on dynamic curriculum settings
      tasks_levels_used: <list> # All the tasks & levels that "can" be used
      sampling_rate: <list> # The rate at which the curriculum is updated in terms of episodes
```
A specific example with values to describe the dynamic curriculum setup

```yaml
  curriculum:
    static:
      toggle: False
      # N/A
    dynamic:
      toggle: True
      tasks_levels_used: [["1","no-traffic"], ["1","low-density"], ["1","mid-density"], ["1","high-density"]]
      sampling_rate: <list> # The rate at which the curriculum is updated in terms of episodes
```
We can use this method to create these single-scenario levels that only generate a specific range of scenarios. So if the user wants to introduce low-density traffic (with any speed) at a given episode or timestep they can "pull" (build) scenarios from that level and used them. Although, an overlying algorithm or policy will automate the process of building and introducing scenarios to the agents. 

Scenarios that have been dynamically created will be stored in a staging directory. The directory will be flushed depending on when the "teacher" or user needs to update the curriculum.

For testing scenarios, the same process applies. However, test scenarios will be only generated once in bulk. You can either specify a test level with desired settings and the numbers of scenarios to be generated in the task configs or specify the number of scenarios and type of scenarios on-the-fly as well. 

- Execute `ultra/train.py`. The following is a list of available arguments for curriculum learning only. To view full list of arguments please refer to [Train and Evaluate a Baseline Agent](docs/getting_started.md).
  - `--curriculum-mode`: Use curriculum learning, will ignore the tasks and levels flag (default is False).
  - `--curriculum-dir`: Path to the curriculum directory which is used to gather task and level information of the grades (default is ../scenarios/curriculum/).
  - `--curriculum-build-scenarios` : Option to automatically build all the scenarios (available for static curriculas) which will be needed from each grade. If you have already build the scenarios then simply ignore this flag (default is False).
  - `--curriculum-scenarios-root-dir` : Specify the directory where the curriculum tasks (config files) are stored (default is ultra/scenarios).
  - `--curriculum-scenarios-save-dir` : Specify the directory to save the scenarios in. Default is to save the scenarios inside it's respective task directory (default is None).


