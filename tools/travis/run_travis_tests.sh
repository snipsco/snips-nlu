#!/usr/bin/env bash

lintingTests=false
sampleTests=false
integrationTest=false

if [[ ${TOXENV} == *py36* ]] && [[ ${TRAVIS_OS_NAME} == "linux" ]]; then
    lintingTests=true
    integrationTest=true

    if [[ ${TRAVIS_BRANCH}  == "master" ]]; then
        sampleTests=true
    fi
fi

./tools/travis/test.sh ${lintingTests} ${integrationTest} ${sampleTests}