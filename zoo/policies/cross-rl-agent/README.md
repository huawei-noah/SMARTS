# cross-rl-agent for behavior model
This provides rl agent for cross scenarios

# install
```bash
cd cross-rl-agent
pip install -e .
```

# run test
```bash
cd test
scl scenario build-all scenarios
python egoless_example.py scenarios/4lane_left_turn
```