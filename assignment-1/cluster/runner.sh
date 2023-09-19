#!/bin/bash

# Set error handling
set -exuo pipefail

# Source environment variables
source src/set_env.sh

# Load singularity module
module load singularity

# Function to check if a command exists
function command_exists() {
  command -v $1 >/dev/null 2>&1
}

# Function to check if an environment variable is set
function env_var_exists() {
    if [ -z "${!1:-}" ]; then
        echo "Error: Environment variable $1 is not set."
        exit 1
    fi
}

# Function to check if a file or directory exists
function path_exists() {
  if [[ ! -e $1 ]]; then
    echo "Error: $1 does not exist."
    exit 1
  fi
}

# Check if required commands are available
command_exists git || { echo "git command not found. Please install git and try again."; exit 1; }
command_exists singularity || { echo "singularity command not found. Please load the singularity module and try again."; exit 1; }

# Check if required environment variables are set
env_var_exists GIT_REPO_NAME
env_var_exists GIT_BRANCH
env_var_exists SLURM_TMPDIR
env_var_exists GITHUB_TOKEN
env_var_exists PRESIGNED_URL
env_var_exists SCRATCH
env_var_exists ARCHIVE

# Turn off shell debugging
set +x

# Get the Git Repo URL
export GIT_REPO_URL=`git config --get remote.origin.url`

# Embed the token into the GIT_REPO_URL
export GIT_TOKEN_URL=$(echo ${GIT_REPO_URL} | sed 's|git@github.com:|https://github.com/|' | sed 's|.git$|.git|' | sed "s|://|://oauth2:${GITHUB_TOKEN}@|")

# Turn on shell debugging
set -x

# Check if the singularity image file exists before trying to copy it
path_exists "${GIT_REPO_NAME}.sif"

# Clone the git repository
echo git clone -b ${GIT_BRANCH} --single-branch ${GIT_REPO_URL} ${SLURM_TMPDIR}/${GIT_REPO_NAME}
set +x
git clone -b ${GIT_BRANCH} --single-branch ${GIT_TOKEN_URL} ${SLURM_TMPDIR}/${GIT_REPO_NAME}
set -x

# Check if the git clone was successful
path_exists "${SLURM_TMPDIR}/${GIT_REPO_NAME}"

# Copy the singularity container to the cloned repo
cp ${GIT_REPO_NAME}.sif ${SLURM_TMPDIR}/${GIT_REPO_NAME}

# Move working directory to $SLURM_TMPDIR
cd ${SLURM_TMPDIR}/${GIT_REPO_NAME}

# Set the dvc cache directory
export DVC_CACHE_DIR=$SCRATCH/${GIT_REPO_NAME}/dvc
export DVC_REMOTE_DIR=$ARCHIVE/${GIT_REPO_NAME}/dvc

# Run singularity command
singularity exec \
  -H $SLURM_TMPDIR/${GIT_REPO_NAME}:/${GIT_REPO_NAME} \
  -B ${DVC_CACHE_DIR} \
  -B ${DVC_REMOTE_DIR} \
  ${GIT_REPO_NAME}.sif \
  bash src/build.sh

# Check if the singularity execution was successful
if [[ $? -ne 0 ]]; then
  echo "Error: Singularity execution failed."
  exit 1
fi
