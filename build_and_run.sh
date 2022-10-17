#!/bin/bash

set -x

sudo docker build --build-arg BASE=latest-gpu -t memory .
sudo docker run --rm -it --runtime nvidia -v=$(pwd)/logs:/workarea/logs memory $@
