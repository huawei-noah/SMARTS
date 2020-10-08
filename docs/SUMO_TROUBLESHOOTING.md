# SUMO Troubleshooting

[[_TOC_]]

## General Disclaimer

SMARTS only officially supports Eclipse SUMO >=1.5.0 and in the platforms currently listed.

## Troubleshooting

### General

You can find a general location for sources and binaries for all platforms here: 
- https://sumo.dlr.de/docs/Downloads.php

If you wish to compile SUMO yourself, the repository is located here: 
 - https://github.com/eclipse/sumo.
 - If you do so make sure to check out the [most current version of 1.7](https://github.com/eclipse/sumo/commits/v1_7_0) or higher.

and the build instructions:  
 - https://sumo.dlr.de/docs/Developer/Main.html#build_instructions

### Linux

SUMO primarily targets Ubuntu versions >= 16.04 So you may not be able to download pre-built binaries for SUMO 1.7 from a package manager if you're running another OS.

If you try through a package manager make sure to command-line call `sumo` to make sure that you have the right version of SUMO.

We would recomment using the prebuilt binaries but if you are using Ubuntu 16 (Xenial),there is a bash script in `extras/sumo/ubuntu_build' that you can use to automate the compilation of SUMO version 1.5.0.

### macOS

macOS installation of SUMO is straight-forward. See https://sumo.dlr.de/docs/Downloads.php#macos_binaries for details.
