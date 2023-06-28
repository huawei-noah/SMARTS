#!/bin/bash

# If HOST_UID is set (and optionally HOST_GID), create and use
# a matching user account inside the container
if [ -n "${HOST_UID}" ]; then
  groupadd --gid "${HOST_GID:-9001}" --force "host-group"
  useradd  --gid "${HOST_GID:-9001}" --uid "${HOST_UID}" --no-log-init --create-home "${HOST_NAME:-'user'}"
  usermod -a -G sudo "${HOST_NAME}"
  PASSWD="1234"
  echo "$HOST_NAME:$PASSWD" | chpasswd
  exec gosu "${HOST_UID}" "$@"
else
  exec "$@"
fi
