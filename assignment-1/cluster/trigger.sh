#!/bin/bash

# Set error handling
set -euo pipefail

source src/set_env.sh

# Check if GITHUB_TOKEN is set. If not, attempt to load it.
if [ -z "${GITHUB_TOKEN:-}" ]
then
    echo "GITHUB_TOKEN is not set. Attempting to load from secrets..."
    source src/set_secrets.sh
fi

# Check if necessary environment variables are set
for var in GIT_REPO_NAME SCRATCH GIT_COMMIT
do
    if [ -z "${!var:-}" ]
    then
        echo "$var is not set. Please set this variable and try again."
        exit 1
    fi
done

# Get current timestamp
TIMESTAMP=`date +%Y%m%d%H%M%S`

# Submit the job
sbatch \
    --export=GITHUB_TOKEN,PRESIGNED_URL \
    --cpus-per-task=1 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --mem-per-cpu=4g \
    --time=12:00:00 \
    --output=${SCRATCH}/${GIT_REPO_NAME}/logs/${TIMESTAMP}_job_%j_%N_${GIT_COMMIT}.log \
    src/runner.sh

# Check for successful job submission
if [ $? -eq 0 ]
then
    echo "Job submitted successfully."
else
    echo "Failed to submit job."
    exit 1
fi
