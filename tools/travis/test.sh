#!/usr/bin/env bash

lintingTests=${1:-false}
sampleTests=${2:-false}
integrationTests=${3:-false}

# Run unittest with tox
python -m unittest discover

# Run linting tests
if [ $lintingTests = true ]; then
    python -m unittest discover -p 'linting_test*.py'
fi

# Run sample tests
if [ $sampleTests = true ]; then
    python -m unittest discover -p 'samples_test*.py'
fi

# Run integration test
if [ $integrationTests = true ]; then
    python -m unittest discover -p 'integration_test*.py'
fi
