# cross-rl-agent for behavior model
This provides rl agent for cross scenarios

# install
```bash
cd cross-rl-agent
pip install -e .
# or 
pip install cross_rl_agent-0.1.0-py3-none-any.whl 
```

# run test
```bash
cd test
scl scenario build-all scenarios
python egoless_example.py scenarios/4lane_left_turn
```