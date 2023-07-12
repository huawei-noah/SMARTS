Please note https://hydra.cc/docs/patterns/configuring_experiments/ for how to generate experiment files.

## 1

In the most simple application the file must start with a global declaration to be relative to the global package

See [Configuration package overrides](https://hydra.cc/docs/advanced/overriding_packages/#defaults-list-package-keywords)
```yaml
# @package _global_
```

## 2
The defaults should select from configuration (or manually specify configuration)

See [Configuration package overrides](https://hydra.cc/docs/advanced/overriding_packages/)
```yaml
# experiment/laner.yaml
defaults:
  - /experiment_default # unnecessary but useful to visualize
  - /agents_configs@agents_configs.agent_black: keep_lane_control-v0 # agent_configs/keep_lane_control-v0.yaml
  - override /env_config: hiway_env-v1_unformatted
  - _self_ # this is also unnecessary because it is implied
```

Note that because the configuration is in the global package an absolute path must be used must be absolute relative to the working directory that was configured from `@hydra.main(config_path=<path>)`.

## 3
Then this experiment can be called like:

```bash
python examples/control.py +experiment=trajectory_tracking +env_config/params/scenarios=intersections
```

See [CLI grammar](https://hydra.cc/docs/advanced/override_grammar/basic/)