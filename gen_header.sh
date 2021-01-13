# !/usr/bin/env bash

# Collate files
python_files="$(find ./cli ./smarts ./benchmark ./envision -name '*.py')"
js_files="$(find ./envision/web/src -name '*.js')"
files="$python_files $js_files"  

# Check and add license notice
for file in $files; do
    if !(grep -q "Copyright (C)" $file); then
        if (echo $file == '*.py'); then
            echo $file
        fi
        #if 
        #sed 's/^/# /' LICENSE | cat - $file >$file.new && mv $file.new $file
    fi
done
