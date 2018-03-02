#!/usr/bin/env bash
if [ "$TRAVIS_OS_NAME" == "osx" ]; then
  python -m pip install --user tox codecov
else
  python -m pip install tox codecov
fi