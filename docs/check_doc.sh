#!/usr/bin/env bash

python docs/write_ontology_doc.py

if [[ `git status --porcelain` ]]; then
  echo "The build step produced some changes that are not versioned"
  git status
  exit 1
fi
