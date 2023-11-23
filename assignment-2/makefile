# This Makefile automates routine tasks for this Singularity-based project.
REPO_NAME := $(shell basename `git rev-parse --show-toplevel` | tr '[:upper:]' '[:lower:]')
IMAGE := $(REPO_NAME).sif
MODEL ?= /network/weights/llama.var/llama2/Llama-2-7b-chat-hf
RUN ?= singularity exec --nv $(FLAGS) -B $(MODEL):$(MODEL) $(IMAGE)
SINGULARITY_ARGS ?=
FLAGS ?=
VENV_PATH ?= venv

.PHONY: report clean jupyter container shell

start_lab:
	mila serve lab --alloc -c 4 --mem=16G -t 6:00:00

start_local:
	salloc --gres=gpu:1 -c 16 --mem=32G -t 6:00:00

loader:
	$(RUN) python3 code/loader.py

repro:
	$(RUN) dvc repro

clean:
	rm -f report/*.blg report/*.fls report/*.out report/*.log report/*.fdb_latexmk report/*.aux report/*.pdf report/*.bbl report/*.toc

jupyter:
	$(RUN)  jupyter lab \
		--ip=0.0.0.0 \
		--no-browser \
		--port 8888

# Builds a Singularity container from the Singularity definition file.
# Note: This command requires sudo privileges.
container: $(IMAGE)
$(IMAGE): Singularity submodules/llama/README.md
	sudo singularity build $(IMAGE) $(SINGULARITY_ARGS) Singularity

submodules/llama/README.md:
	git submodule init && git submodule update

REMOTE ?= cn-f001
push:
	rsync -rvahzP ${IMAGE} ${REMOTE}.server.mila.quebec:comp-550/assignment-2/

# Starts a shell within the Singularity container, with the virtual environment activated.
shell:
	singularity shell --nv $(FLAGS) -B $(MODEL) $(IMAGE) $(SINGULARITY_ARGS) bash
