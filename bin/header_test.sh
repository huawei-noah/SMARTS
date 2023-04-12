# !/usr/bin/env bash

# Collate all files excluding auto generated files of *_pb2.py and *_pb2_grpc.py.
# We also exlude files in the waymo_open_dataset module.
python_files="$(find ./baselines/marl_benchmark ./cli ./envision ./smarts -name '*.py' ! -name '*_pb2.py' ! -name '*_pb2_grpc.py' ! -wholename './smarts/waymo/waymo_open_dataset/__init__.py' ! -wholename './smarts/waymo/waymo_open_dataset/protos/__init__.py')"
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
else
    echo "Header test passed"
fi
