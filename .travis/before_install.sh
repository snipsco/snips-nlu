#!/usr/bin/env bash
# Install Rust
curl https://sh.rustup.rs -sSf | bash -s -- -y
export PATH="/usr/local/bin:$HOME/.cargo/bin:$PATH"
if [ "$TRAVIS_OS_NAME" == "osx" ]; then
  brew update || brew update

  brew outdated openssl || brew upgrade openssl
  brew install openssl@1.1

  # install pyenv
  git clone --depth 1 https://github.com/pyenv/pyenv ~/.pyenv
  PYENV_ROOT="$HOME/.pyenv"
  PATH="$PYENV_ROOT/bin:$PATH"
  eval "$(pyenv init -)"

  case "${TOXENV}" in
    py27)
      pyenv install 2.7.14
      pyenv global 2.7.14
      ;;
    py36)
      pyenv install 3.6.4
      pyenv global 3.6.4
      ;;
  esac
  pyenv rehash
  # A manual check that the correct version of Python is running.
  python --version
fi