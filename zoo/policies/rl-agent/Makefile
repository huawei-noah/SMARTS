.PHONY: format
format:
	# pip install black==19.10b0
	black .

.PHONY: rm-pycache
rm-pycache:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

.PHONY: wheel
wheel:
	python setup.py clean --all
	python setup.py bdist_wheel
	@echo
	@echo "ls ./dist:"
	@ls -l1 ./dist | sed 's/^/  /'

.PHONY: install
install:
	pip install -e .
