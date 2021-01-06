#!/bin/bash
set -eo pipefail

# Run XDummy
/usr/bin/Xorg -noreset +extension GLX +extension RANDR +extension RENDER -logfile ./xdummy.log -config /etc/X11/xorg.conf $DISPLAY &

# Execute CMD
exec "$@"
