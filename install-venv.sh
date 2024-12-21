#!/usr/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

python3 -m venv $SCRIPT_DIR/venv

$SCRIPT_DIR/venv/bin/pip install -r $SCRIPT_DIR/requirements.txt
