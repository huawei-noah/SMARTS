#!/usr/bin/env bash

for policy in ./policies/*
do
    if [ -d "${policy}/tests" ]; then
        echo "Testing ${policy}"
        venv_dir=`mktemp -u -d`
        python3.7 -m venv "${venv_dir}"
        source "${venv_dir}/bin/activate"
        whl="$(find ${policy} | grep .whl | sort -n | tail -n1)"
        echo "Installing SMARTS"
        pip install -e ../../../ # install smarts
        echo "Installing ${whl}"
        pip install "${whl}" # install the agent
        pip install pytest
        cd $policy && pytest -v -s -o log_cli=1 -o log_cli_level=INFO
        deactivate # exit the venv
    fi
done
