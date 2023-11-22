# This Makefile automates routine tasks for this Singularity-based project.
REPO_NAME := $(shell basename `git rev-parse --show-toplevel` | tr '[:upper:]' '[:lower:]')
IMAGE := $(REPO_NAME).sif
RUN ?= singularity exec $(FLAGS) $(IMAGE)
SINGULARITY_ARGS ?=
FLAGS ?=
VENV_PATH ?= venv

.PHONY: report clean jupyter container shell

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

# Starts a shell within the Singularity container, with the virtual environment activated.
shell:
	singularity shell $(FLAGS) $(IMAGE) $(SINGULARITY_ARGS) bash
