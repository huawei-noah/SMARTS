## Creating Scenarios

<img align="right" width="400" height="400" src="research/edmonton/media/intersections/stopwatcher.gif"/>

### Summary of contents:
- **pool/** : 
  - contains different maps with various number of lanes [2,3,4,5,6], 
  - max speeds [ 50kmh, 70kmh, 100kmh]
  - shapes [c: cross-intersection, t: t-intersection]
- **common/**: 
  - `distributions.py`: script for defining different traffic patterns 
  - `social_vehicle_definitions`: contains different social behaviors
  - `visualization.py`: a tool for visualizing scenarios
- **task1/**: 
  - `config.yaml`: define ego-mission and traffic-patterns:
  
### How to use:
- 1- generate maps : `$ scl scenario build-all research/edmonton/intersections/scenarios/pool/`
- 2- define task in config.yaml:
```yaml
  ego_missions: 
  # can define multiple missions
  # (example: left turn from south to west)
  - start: south-SN
    end:   west-EW
 levels:
  easy:
    train:
      total: 800
      intersection_types:
        2lane_t: # train on t-intersections
           percent: 0.5
           specs: [[50kmh,low-density,0.33],[70kmh,low-density,0.33],[100kmh,low-density,0.33]]
        3lane_t:
           percent: 0.5
           specs: [[50kmh,low-density,0.33],[70kmh,low-density,0.33],[100kmh,low-density,0.33]]
    test:
      total: 200
      intersection_types:
        2lane_c: # test on cross-intersections
           percent: 0.5
           specs:  [[50kmh,low-density,0.33],[70kmh,low-density,0.33],[100kmh,low-density,0.33]]
        3lane_c:
           percent: 0.5
           specs: [50kmh,low-density,0.33],[70kmh,low-density,0.33],[100kmh,low-density,0.33]]
  hard:
    # same pattern with different densities
``` 
- 3- generate scenarios for training: `$ python research/edmonton/intersections/scenarios/interface.py generate --task 1 --level easy --num 5`
  - --task refers to the config.yaml file
  - --level selects the level from config.yaml
  - --num is the number of seeds to generate flows based on (for 5 seeds it generate the same scenario with 5 different seeds)
  - for each scenario it generates `metadata.json` to show the routes and number/types of vehicle

### Analyzing and measuring max_steps
- 1- generate scenarios for analysis: `$ python research/edmonton/intersections/scenarios/interface.py generate --task 1 --level easy --num 5 --stopwatcher aggressive` 
  - other types of stopwatchers are available [default, slow, blocker, crusher] and the path is always from south to west
- 2- analyze/visualize: 
  - `$ python research/edmonton/intersections/scenarios/interface.py analyze --scenarios research/edmonton/intersections/scenarios/task1/{analysis scenarios} --max-steps 6000 --video 100 --output {path to output}`
  - runs the scenario for 6000 steps and records all vehicle states for analysis
  - saves the analysis.pkl in output directory
  - if --video 100 is passed, every 100 scenarios it saves a gif of that scenario; (video saving is very time consuming avoid using it for long runs)
- 3- analyze in-junction behaviors
  - `$ python research/edmonton/intersections/scenarios/interface.py in-junction --scenarios research/edmonton/intersections/scenarios/task1/{analysis scenarios} --video --output {path to output}`
