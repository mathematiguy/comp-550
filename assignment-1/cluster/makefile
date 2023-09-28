# This Makefile automates routine tasks for this Singularity-based project.
REPO_NAME := $(shell basename `git rev-parse --show-toplevel` | tr '[:upper:]' '[:lower:]')
IMAGE := $(REPO_NAME).sif
RUN ?= singularity exec $(FLAGS) -B submodules/llama:/opt/llama $(IMAGE)
SINGULARITY_ARGS ?=
DVC_CACHE_DIR ?= $(shell dvc cache dir)
FLAGS ?=
VENV_PATH ?= venv
ARCHIVE ?= /archive
SCRATCH ?= /scratch

.PHONY: show_logs trigger scratch archive repro predict start jupyter container push shell

gpu_alloc:
	salloc --gres=gpu:1 -c 4 --mem=10G -t 1:00:00

show_logs:
	ls ${SCRATCH}/${REPO_NAME}/logs/*.log | tail -n1 | xargs cat

set_secrets:
	bash -c "./src/set_secrets.sh && echo $$${GITHUB_TOKEN}"

encrypt: src/secrets.yaml.enc
src/secrets.yaml.enc: src/secrets.yaml
	openssl aes-256-cbc -a -salt -in $< -out $@ -pbkdf2

squeue:
	squeue -u caleb.moses

trigger:
	bash src/trigger.sh

# Use this command to send the singularity container to a running remote session on the cluster
push: USER_NAME=caleb.moses
push: SERVER=cn-f001
push: OBJECT=$(IMAGE)
push: REMOTE=$(USER_NAME)@$(SERVER).server.mila.quebec
push: DEST=$(REPO_NAME)/
push:
	rsync -ahP $(OBJECT) $(REMOTE):$(DEST)

dvc-setup: $(SCRATCH)/$(REPO_NAME)/logs \
	$(SCRATCH)/$(REPO_NAME)/dvc \
	$(ARCHIVE)/$(REPO_NAME)/dvc

$(SCRATCH)/$(REPO_NAME)/logs:
	mkdir -p $(SCRATCH)/$(REPO_NAME)/logs

$(SCRATCH)/$(REPO_NAME)/dvc:
	mkdir -p $(SCRATCH)/$(REPO_NAME)/dvc
	dvc cache dir --local $(SCRATCH)/$(REPO_NAME)/dvc

$(ARCHIVE)/$(REPO_NAME)/dvc:
	mkdir -p $(ARCHIVE)/$(REPO_NAME)/dvc
	dvc remote add -d archive $(ARCHIVE)/$(REPO_NAME)/dvc
