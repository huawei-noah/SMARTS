.PHONY: test
test: build-all-scenarios
	# sstudio uses hash(...) as part of some of its type IDs. To make the tests
	# repeatable we fix the seed.
	PYTHONPATH=$(PWD) PYTHONHASHSEED=42 pytest -v \
		--cov=smarts \
		--doctest-modules \
		--forked \
		--dist=loadscope \
		-n `expr \( \`nproc\` \/ 2 \& \`nproc\` \> 3 \) \| 2` \
		--nb-exec-timeout 65536 \
		./examples/tests ./smarts/env ./envision ./smarts/core ./smarts/sstudio ./tests \
		--ignore=./smarts/core/waymo_map.py \
		--ignore=./smarts/core/argoverse_map.py \
		--ignore=./smarts/core/tests/test_argoverse.py \
		--ignore=./smarts/core/tests/test_smarts_memory_growth.py \
		--ignore=./smarts/core/tests/test_env_frame_rate.py \
		--ignore=./smarts/core/tests/test_notebook.py \
		--ignore=./smarts/env/tests/test_benchmark.py \
		--ignore=./examples/tests/test_learning.py \
		-k 'not test_long_determinism'
	rm -f .coverage.*
	rm -f .coverage*

.PHONY: sanity-test
sanity-test: build-sanity-scenarios
	PYTHONPATH=$(PWD) PYTHONHASHSEED=42 pytest -v \
		--doctest-modules \
		--forked \
		--dist=loadscope \
		--junitxml="sanity_test_result.xml" \
		-n `expr \( \`nproc\` \/ 2 \& \`nproc\` \> 3 \) \| 2` \
		./smarts/core/tests/test_python_version.py::test_python_version \
		./smarts/core/tests/test_sumo_version.py::test_sumo_version \
		./smarts/core/tests/test_dynamics_backend.py::test_set_pose \
		./smarts/core/tests/test_sensors.py::test_waypoints_sensor \
		./smarts/core/tests/test_smarts.py::test_smarts_doesnt_leak_tasks_after_reset \
		./examples/tests/test_examples.py::test_examples[hiway_v1] \
		./examples/tests/test_examples.py::test_examples[laner] \
		./smarts/env/tests/test_social_agent.py::test_social_agents_not_in_env_obs_keys

.PHONY: test-learning
test-learning: build-all-scenarios
	pytest -v -s -o log_cli=1 -o log_cli_level=INFO ./examples/tests/test_learning.py

.PHONY: test-memory-growth
test-memory-growth: build-all-scenarios
	PYTHONHASHSEED=42 pytest -v \
		--cov=smarts \
		--forked \
		--dist=loadscope \
		-n `nproc --ignore 1` \
		./smarts/core/tests/test_smarts_memory_growth.py
	rm -f .coverage.*
	rm -f .coverage*

.PHONY: test-long-determinism
test-long-determinism:
	scl scenario build --clean scenarios/sumo/minicity
	PYTHONHASHSEED=42 pytest -v \
		--forked \
		./smarts/env/tests/test_determinism.py::test_long_determinism

.PHONY: benchmark
benchmark: build-all-scenarios
	pytest -v ./smarts/env/tests/test_benchmark.py

.PHONY: test-zoo
test-zoo: build-all-scenarios
	cd smarts/zoo/policies && make test

.PHONY: run
run: build-scenario
	python $(script) $(scenario)

.PHONY: build-all-scenarios
build-all-scenarios:
	scl scenario build-all scenarios

.PHONY: build-sumo-scenarios
build-sumo-scenarios:
	scl scenario build-all scenarios/sumo

.PHONY: build-sanity-scenarios
build-sanity-scenarios:
	scl scenario build --clean scenarios/sumo/loop
	scl scenario build --clean scenarios/sumo/zoo_intersection

.PHONY: build-scenario
build-scenario:
	scl scenario build $(scenario)

.PHONY: flamegraph
flamegraph: $(scenario)/flamegraph.svg
	# We want the default browser to open this flamegraph since it's an
	# interactive SVG, browsers will accept anything you throw at
	# them with a .html extension so lets just rename this svg so that
	# `open` will open the file with the browser.
	mv $(scenario)/flamegraph.svg $(scenario)/flamegraph.html

	# Python's webbrowser module is a cross-platform way of opening a webpage using a preferred browser
	python -m webbrowser $(scenario)/flamegraph.html

.PHONY: $(scenario)/flamegraph.svg
$(scenario)/flamegraph.svg: $(scenario)/flamegraph-perf.log
	./utils/third_party/tools/flamegraph.pl --title "HiWay CPU ($(scenario))" $(scenario)/flamegraph-perf.log > $(scenario)/flamegraph.svg

.PHONY: $(scenario)/flamegraph-perf.log
$(scenario)/flamegraph-perf.log: build-scenario $(script) smarts/core/* smarts/env/*
	# pip install git+https://github.com/asokoloski/python-flamegraph.git
	python -m flamegraph -i 0.001 -o $(scenario)/flamegraph-perf.log $(script) $(scenario)

.PHONY: sumo-gui
sumo-gui: $(scenario)/map.net.xml
	sumo-gui \
		-n ./$(scenario)/map.net.xml

.PHONY: header-test
header-test:
	bash ./bin/header_test.sh

.PHONY: gen-header
gen-header:
	bash ./bin/gen_header.sh

.PHONY: clean
clean:
	# we use `rm -f` for discard errors when the file does not exist
	rm -f ./$(scenario)/map.egg
	rm -f ./$(scenario)/map.glb
	rm -f ./$(scenario)/bubbles.pkl
	rm -f ./$(scenario)/missions.pkl
	rm -f ./$(scenario)/friction_map.pkl
	rm -f ./$(scenario)/flamegraph-perf.log
	rm -f ./$(scenario)/flamegraph.svg
	rm -f ./$(scenario)/flamegraph.html
	rm -f ./$(scenario)/*.rou.xml
	rm -f ./$(scenario)/*.rou.alt.xml
	rm -f ./$(scenario)/*.smarts.xml
	rm -rf ./$(scenario)/traffic
	rm -rf ./$(scenario)/social_agents
	rm -f .coverage.*
	rm -f .coverage*

.PHONY: format
format:
	echo "isort, version `isort --version-number`"
	isort -m VERTICAL_HANGING_INDENT --skip-gitignore --ac --tc --profile black ./baselines ./cli ./envision ./examples/ ./utils/ ./scenarios/ ./smarts ./zoo
	black --version
	black .
	# npm install prettier
	# Make sure to install Node.js 14.x before running `prettier`
	npx prettier --write envision/web/src

.PHONY: docs
docs:
	cd docs && make clean html

.PHONY: wheel
wheel: docs
	python setup.py clean --all
	python setup.py bdist_wheel
	@echo
	@echo "ls ./dist:"
	@ls -l1 ./dist | sed 's/^/  /'

.PHONY: rm-pycache
rm-pycache:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

.PHONY: rm-cov
rm-cov:
	find . -type f -name ".coverage.*" -delete
	find . -type f -name ".coverage*" -delete

