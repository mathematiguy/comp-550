#!/bin/bash

# Show commands (-e) and end script on fail (-x)
set -ex

# Source environment variables
source src/set_env.sh

# Create a new branch for this job
export JOB_NAME="job_${SLURM_JOB_ID}"
git checkout -b ${JOB_NAME} || { echo "Failed to create new branch. Exiting."; exit 1; }

# Error and exit handling
trap 'handle_exit' EXIT

handle_exit() {
  local exit_code=$?
  if [ $exit_code -eq 0 ]; then
    echo "Job succeeded."
  else
    echo "Job failed or was terminated. Please check the branch ${JOB_NAME} for details."
  fi

  # Merge and cleanup job branch
  git checkout main
  git diff --quiet ${JOB_NAME} main || {
    git merge --no-ff ${JOB_NAME} || git push origin ${JOB_NAME} && echo "Failed to merge. Please merge manually."
    git branch -d ${JOB_NAME} || echo "Failed to delete the branch. Please delete manually."
  }

  exit $exit_code
}

# Load stages from dvc.yaml
# stages=$(yq e '(.stages | keys)[]' dvc.yaml)
stages="download_models"

# Load models from params.yaml in a reproducible order
models=$(yq e '(.model | keys)[]' params.yaml) # | sort -R --random-source=<(printf "%1000s" "12345"))

# Set the dvc cache directory
dvc cache dir --local ${DVC_CACHE_DIR}

# Pull dvc cache from archive
dvc remote add archive ${DVC_REMOTE_DIR}

# Checkout dvc.lock
dvc checkout || true

for model in $models; do
    for stage in $stages; do
        full_stage_name="$stage@$model"
        echo "Running stage: $full_stage_name"
        dvc repro -s $full_stage_name
    done
    echo "Committing dvc.lock and pushing changes to Git if any changes..."
    git add dvc.lock
    git diff --cached --quiet || {
        git commit -m "Update dvc.lock for model $model by ${JOB_NAME}"
        git push -u origin ${JOB_NAME}
    }
done

# Push to archive remote
dvc push -r archive

echo ${JOB_NAME} completed at `date +"%m-%d-%Y %H:%M:%S"`
