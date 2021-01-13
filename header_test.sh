# !/usr/bin/env bash

# Collate files
python_files="$(find ./cli ./smarts ./benchmark ./envision -name '*.py')"
js_files="$(find ./envision/web/src -name '*.js')"
files="$python_files $js_files"  

# Check if file has license notice
exit_code=0
for file in $files; do
    if !(grep -q "Copyright (C)" $file); then
        echo "no copyright header in $file" >&2
        exit_code=1
    fi
done

if [ $exit_code = 1 ]; then
    exit 1
fi
