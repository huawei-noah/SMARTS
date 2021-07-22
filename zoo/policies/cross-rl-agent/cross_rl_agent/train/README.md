## Build scenarios
To build the scenarios run:
```bash
# cd zoo/policies/cross-rl-agent/cross_rl_agent/train
$ scl scenario build-all scenarios
```

## Open envision
To start the envision server run the following:
```bash
# cd zoo/policies/cross-rl-agent/cross_rl_agent/train
$ scl envision start -s scenarios
```
and open `localhost:8081` in your local browser.

## Run simple keep lane example
To run an example run:
```bash
# cd zoo/policies/cross-rl-agent/cross_rl_agent/train
$ python3.7 run_test.py scenarios/4lane_left_turn
```


## Run train example 
To train an agent:
```bash
# cd zoo/policies/cross-rl-agent/cross_rl_agent/train
$ python3.7 run_train.py scenarios/4lane_left_turn #--headless
```
For fast training, you can stop the envision server and add `--headless`.
