#!/usr/bin/env bash

warn() { echo "$@" >&2; }

die() { warn "$@"; exit 1; }
