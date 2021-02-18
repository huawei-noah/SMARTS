.PHONY: test
test: build-all-scenarios
	# sstudio uses hash(...) as part of some of its type IDs. To make the tests
	# repeatable we fix the seed.
	PYTHONHASHSEED=42 pytest -v \
		--cov=smarts \
		--doctest-modules \
		--forked \
		--dist=loadscope \
		-n `nproc --ignore 1` \
		./envision ./smarts/contrib ./smarts/core ./smarts/env ./smarts/sstudio ./tests \
		--ignore=./smarts/core/tests/test_smarts_memory_growth.py \
		--ignore=./smarts/env/tests/test_benchmark.py \
		--ignore=./smarts/env/tests/test_learning.py \
		-k 'not test_long_determinism'

.PHONY: test-learning
test-learning: build-all-scenarios
	pytest -v -s -o log_cli=1 -o log_cli_level=INFO ./smarts/env/tests/test_learning.py

.PHONY: test-memory-growth
test-memory-growth: build-all-scenarios
	PYTHONHASHSEED=42 pytest -v \
		--cov=smarts \
		--forked \
		--dist=loadscope \
		-n `nproc --ignore 1` \
		./smarts/core/tests/test_smarts_memory_growth.py

.PHONY: test-long-determinism
test-long-determinism: 
	scl scenario build --clean scenarios/minicity
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
	./third_party/tools/flamegraph.pl --title "HiWay CPU ($(scenario))" $(scenario)/flamegraph-perf.log > $(scenario)/flamegraph.svg

.PHONY: $(scenario)/flamegraph-perf.log
$(scenario)/flamegraph-perf.log: build-scenario $(script) smarts/core/* smarts/env/*
	# pip install git+https://github.com/asokoloski/python-flamegraph.git
	python -m flamegraph -i 0.001 -o $(scenario)/flamegraph-perf.log $(script) $(scenario)

.PHONY: pview
pview: $(scenario)/map.egg
	# !!! READ THE pview MANUAL !!!
	# https://www.panda3d.org/manual/?title=Previewing_3D_Models_in_Pview
	pview -c -l $(scenario)/map.egg

.PHONY: sumo-gui
sumo-gui: $(scenario)/map.net.xml
	sumo-gui \
		-n ./$(scenario)/map.net.xml

.PHONY: header-test
header-test:
	bash ./header_test.sh

.PHONY: gen-header
gen-header:
	bash ./gen_header.sh

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
	rm -rf ./$(scenario)/traffic
	rm -rf ./$(scenario)/social_agents

.PHONY: format
format:
	# pip install isort==5.7.0
	isort -m VERTICAL_HANGING_INDENT --skip-gitignore --ac --tc --profile black ./benchmark/ ./cli ./envision ./examples/ ./extras/ ./scenarios/ ./smarts ./ultra ./zoo
	# pip install black==20.8b1
	black .
	# npm install prettier
	# Make sure to install Node.js 14.x before running `prettier`
	npx prettier --write envision/web/src

.PHONY: docs
docs:
	(cd docs; make clean html)

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
