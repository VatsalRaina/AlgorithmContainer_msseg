#!/usr/bin/env bash

./build.sh

docker save baseline | gzip -c > Baseline.tar.gz
