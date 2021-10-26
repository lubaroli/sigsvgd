#!/bin/sh

set -x
set -e

git submodule init
git submodule update --init --recursive

cmake -Bbuild || exit 1
make -Cbuild || exit 1
