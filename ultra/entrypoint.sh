#!/bin/bash
set -eo pipefail
export DISPLAY=:1
# Run XDummy

/usr/bin/Xorg -noreset +extension GLX +extension RANDR +extension RENDER -logfile ./xdummy.log -config /etc/X11/xorg.conf -novtswitch $DISPLAY &
# Execute CMD
exec "$@"
