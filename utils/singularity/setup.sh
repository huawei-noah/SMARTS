#!/usr/bin/env bash

# Suppress message of missing /dev/input folder 
mkdir -p /dev/input

# Copy and paste smarts.egg-info if not available
if [[ ! -d /src/smarts.egg-info ]]; then
    cp -r /media/smarts.egg-info /src/smarts.egg-info;
    chmod -R 777 /src/smarts.egg-info;
fi
