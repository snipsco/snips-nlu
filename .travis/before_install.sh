#!/usr/bin/env bash
if [ "$TRAVIS_OS_NAME" == "osx" ]; then
  # install pyenv
  git clone --depth 1 https://github.com/pyenv/pyenv ~/.pyenv
  PYENV_ROOT="$HOME/.pyenv"
  PATH="$PYENV_ROOT/bin:$PATH"
  eval "$(pyenv init -)"

  case "${TOXENV}" in
    integration-test)
      pyenv install 3.6.4
      pyenv global 3.6.4
      ;;
  esac
  pyenv rehash
  # A manual check that the correct version of Python is running.
  python --version
fi