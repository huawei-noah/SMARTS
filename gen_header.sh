# !/usr/bin/env bash

files=""

if [[ $# -eq 0 ]]; then
    # No specific file specified collate all files
    python_files="$(find ./cli ./smarts ./benchmark ./envision -name '*.py')"
    js_files="$(find ./envision/web/src -name '*.js')"
    files="$python_files $js_files"  
else
    # use files specified in the args
    for file in "$@"; do
        files="$files $file"
    done 
fi 

# Check and add license notice
for file in $files; do
    if !(grep -q "Copyright (C)" $file); then
        if [[ $file == *'.py' ]]; then
            sed 's/^/# /' LICENSE | cat - $file >$file.new && mv $file.new $file
        else
            sed 's/^/\/\/ /' LICENSE | cat - $file >$file.new && mv $file.new $file
        fi
        echo "Added copyright header in $file"
    fi
done
