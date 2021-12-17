#!/usr/bin/env bash

function check_pylint_existance {
    if [ $(dpkg-query -W -f ='${Status}' pylint 2>/dev/null | grep -c "pylint installed") -eq 0 ];
    then
        echo "Installing pylint (used for code formatting)..."
        pip install pylint
    fi
}
function run_pylint {
    echo "Checking docstring existance for all files in the current directory and all sub-directories..."
    # Please check https://pylint.pycqa.org/en/latest/index.html for manual
    pylint -d all -e missing-docstring -s n --msg-template='{path}: {msg_id}: line {line}: {msg} ({symbol})' --output=./utils/setup/docstring_check_result.txt *
    echo "Done."
}

function main {
    check_pylint_existance
    run_pylint
}

main