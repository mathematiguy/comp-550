# This Makefile automates routine tasks for this Singularity-based project.
REPO_NAME := $(shell basename `git rev-parse --show-toplevel` | tr '[:upper:]' '[:lower:]')
IMAGE := $(REPO_NAME).sif
RUN ?= singularity exec $(FLAGS) -B submodules/llama:/opt/llama $(IMAGE)
SINGULARITY_ARGS ?=
DVC_CACHE_DIR ?= $(shell dvc cache dir)
FLAGS ?=
VENV_PATH ?= venv

include cluster/makefile

.PHONY: show_logs trigger scratch archive repro predict start jupyter container push shell

test_llama:
	$(RUN) torchrun --nproc_per_node 1 submodules/llama/example_chat_completion.py \
		--ckpt_dir models/llama-2-7b-chat/ \
		--tokenizer_path models/llama/tokenizer.model \
		--max_seq_len 512 --max_batch_size 6

convert_llama_weights:
	$(RUN) python3 submodules/transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir models/llama-2-7b --model_size 7B --output_dir hf_models/llama-2-7b

jupyter:
	$(RUN)  jupyter lab \
		--ip=0.0.0.0 \
		--no-browser \
		--port 8888

# Builds a Singularity container from the Singularity definition file.
# Note: This command requires sudo privileges.
container: $(IMAGE)
$(IMAGE): Singularity
	sudo singularity build $(IMAGE) $(SINGULARITY_ARGS) Singularity

# Use this command to send the singularity container to a running remote session on the cluster
push: USER_NAME=caleb.moses
push: SERVER=cn-f001
push: OBJECT=$(IMAGE)
push: REMOTE=$(USER_NAME)@$(SERVER).server.mila.quebec
push: DEST=$(REPO_NAME)/
push:
	rsync -ahP $(OBJECT) $(REMOTE):$(DEST)

# Starts a shell within the Singularity container, with the virtual environment activated.
shell:
	singularity shell $(FLAGS) $(IMAGE) $(SINGULARITY_ARGS) bash
