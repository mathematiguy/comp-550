#!/bin/bash

# Set environment variables
export GIT_REPO_NAME=$(basename `git rev-parse --show-toplevel` | tr '[:upper:]' '[:lower:]')
export GIT_BRANCH=`git rev-parse --abbrev-ref HEAD`
export GIT_COMMIT=`git rev-parse --short HEAD`

# Echo the variables to verify they've been set
echo "GIT_REPO_NAME=$GIT_REPO_NAME"
echo "GIT_BRANCH=$GIT_BRANCH"
echo "GIT_COMMIT=$GIT_COMMIT"
