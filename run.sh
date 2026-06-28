#!/usr/bin/env bash
set -u -o pipefail
cd /root/task

pip install -q -r requirements.txt

python -m agent --selfcheck
if [ $? -ne 0 ]; then
  echo "Selfcheck failed" >&2
  exit 1
fi

python -m pytest -q; rc=$?
if [ "$rc" -le 1 ]; then
  echo "ready"
  exit 0
else
  exit "$rc"
fi
