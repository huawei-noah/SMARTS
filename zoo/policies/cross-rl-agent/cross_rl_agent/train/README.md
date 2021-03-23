# generate scenario
run
```bash
$ scl scenario build-all scenarios
```

# open envision
run
```bash
$ scl envision start -s scenarios
```
and open `localhost:8081` in your local browser

# run simple keep lane example
run
```bash
$ python run_keeplane.py scenarios/4lane_left_turn
```
and refresh the browser



# run train example 
run
```bash
$ python run_train.py scenarios/4lane_left_turn
```
and refresh the browser,
for fast training, you can turn off envision by pass param `headless=True`
into env
